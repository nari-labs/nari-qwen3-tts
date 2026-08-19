from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from model_optimization.rows import talker_decode_rows, talker_decode_static, talker_prefill_static
from nari_qwen3_tts.contract import TalkerDecodeCaptureKey, TalkerPrefillCaptureKey
from nari_qwen3_tts.executor import (
    CudaGraphPoolFence,
    SlotBusyError,
    TalkerDecodeExecutionRow,
    TalkerDecodeRowsExecutionInput,
    TalkerPrefillExecutionRow,
    TalkerPrefillRowsExecutionInput,
    TalkerSamplingExecutionRow,
)
from nari_qwen3_tts.executor import (
    TalkerExecutor as TalkerCudaExecutor,
)
from nari_qwen3_tts.executor.types import TalkerDecodeInput, TalkerSamplingInput


@dataclass
class _Captured:
    operation: object

    def replay(self):
        return self.operation()


class _Driver:
    def capture(self, operation, *, after_warmup=None):
        operation()
        if after_warmup is not None:
            after_warmup()
        return _Captured(operation)


@dataclass
class _PersistentCaptured:
    operation: object
    output: object

    def replay(self):
        fresh = self.operation()
        assert isinstance(fresh, tuple) and isinstance(self.output, tuple)
        for destination, source in zip(self.output, fresh, strict=True):
            destination.copy_(source)
        return self.output


class _PersistentDriver:
    def capture(self, operation, *, after_warmup=None):
        output = operation()
        if after_warmup is not None:
            after_warmup()
        return _PersistentCaptured(operation, output)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        values = torch.arange(48, dtype=torch.float32).reshape(16, 3) / 10
        self.embedding = torch.nn.Embedding.from_pretrained(values, freeze=True)
        self.projected = values.flip(1).contiguous()
        self.initialized = False

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def initialize_projected_text_embedding_cache(self) -> None:
        self.initialized = True

    def get_projected_text_embedding_cache(self) -> torch.Tensor:
        return self.projected

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(self, *, input_embeds: torch.Tensor, attention_context: object) -> torch.Tensor:
        del attention_context
        return input_embeds

    @staticmethod
    def codec_head(hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat((hidden, hidden.sum(1, keepdim=True)), dim=1)


def _config() -> object:
    predictor = SimpleNamespace(vocab_size=4)
    talker = SimpleNamespace(
        hidden_size=3,
        vocab_size=4,
        codec_eos_token_id=3,
        code_predictor=predictor,
    )
    return SimpleNamespace(talker=talker, code_predictor=predictor)


class _Publication:
    def __init__(self, request_id):
        self.request_id = request_id


class _Context:
    pass


class _Cache:
    def __init__(self):
        self.prepared = []

    def create_decode(self, key, *, slot):
        return _Context()

    def create_prefill(self, key, *, slot):
        return _Context()

    def prepare_capture(self, context, key):
        self.prepared.append(("capture", context, key))

    def prepare_decode(self, context, key, request_ids, *, reuse_attention_plan=False):
        self.prepared.append(("decode", context, key, request_ids, reuse_attention_plan))
        return tuple(_Publication(request_id) for request_id in request_ids)

    def prepare_prefill(self, context, key, request_ids, sequence_lengths):
        self.prepared.append(("prefill", context, key, request_ids, sequence_lengths))
        return tuple(_Publication(request_id) for request_id in request_ids)

    @staticmethod
    def abort(publications):
        del publications


class _AbortRecordingCache(_Cache):
    def __init__(self):
        super().__init__()
        self.aborted = []

    def abort(self, publications):
        self.aborted.append(tuple(publication.request_id for publication in publications))


class _FailOnceCaptured:
    def __init__(self, successor):
        self.successor = successor
        self.failed = False

    def replay(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected replay failure")
        return self.successor.replay()


def _direct_sampling_row() -> TalkerSamplingExecutionRow:
    return TalkerSamplingExecutionRow(
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        seed=7,
        offset=512,
        seen_token_mask=torch.zeros(4, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", float("nan")),
        ("top_k", -1),
        ("top_k", True),
        ("top_p", 0.0),
        ("repetition_penalty", float("inf")),
        ("seed", -1),
        ("offset", -1),
    ],
)
def test_direct_talker_sampling_rejects_invalid_logical_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        replace(_direct_sampling_row(), **{field: value})


def test_direct_talker_rows_validate_tensor_contracts_before_staging() -> None:
    sampling = _direct_sampling_row()
    with pytest.raises(TypeError, match="seen_token_mask"):
        replace(sampling, seen_token_mask=torch.zeros(4))
    with pytest.raises(TypeError, match="text_token_ids"):
        TalkerPrefillExecutionRow(
            text_token_ids=torch.ones(2),
            codec_token_ids=torch.ones(2, dtype=torch.long),
            codec_token_mask=torch.ones(2, dtype=torch.bool),
            suppress_eos=True,
            sampling=sampling,
        )
    with pytest.raises(ValueError, match="talker_step_embed"):
        TalkerDecodeExecutionRow(
            talker_step_embed=torch.ones((1, 4)),
            text_token_id=torch.ones((), dtype=torch.long),
            suppress_eos=True,
            sampling=sampling,
        )


def test_talker_decode_row_allows_host_text_for_pinned_batch_staging() -> None:
    row = TalkerDecodeExecutionRow(
        talker_step_embed=torch.ones(4, device="meta"),
        text_token_id=torch.tensor([3], dtype=torch.long),
        suppress_eos=True,
        sampling=replace(_direct_sampling_row(), seen_token_mask=None),
    )

    assert row.text_token_id.device.type == "cpu"


def _values(order: tuple[int, ...], *, rows: int = 3) -> TalkerDecodeInput:
    step = torch.tensor([[5.0, 1, 0], [0.0, 2, 7], [1.0, 9, 2]])[:rows]
    text = torch.tensor([0, 1, 2])[:rows]
    suppress = torch.tensor([False, True, False])[:rows]
    sampling = TalkerSamplingInput(
        temperature=torch.zeros(rows),
        top_k=torch.ones(rows, dtype=torch.int32),
        top_p=torch.ones(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.tensor([3, 11, 29], dtype=torch.long)[:rows],
        offsets=torch.tensor([0, 32, 64], dtype=torch.long)[:rows],
        seen_token_mask=torch.zeros((rows, 4), dtype=torch.bool),
    )
    index = torch.tensor(order, dtype=torch.long)
    return TalkerDecodeInput(
        attention_context=None,
        talker_step_embed=step.index_select(0, index),
        text_token_ids=text.index_select(0, index),
        suppress_eos=suppress.index_select(0, index),
        sampling=TalkerSamplingInput(
            temperature=sampling.temperature.index_select(0, index),
            top_k=sampling.top_k.index_select(0, index),
            top_p=sampling.top_p.index_select(0, index),
            repetition_penalty=sampling.repetition_penalty.index_select(0, index),
            seed=sampling.seed.index_select(0, index),
            offsets=sampling.offsets.index_select(0, index),
            seen_token_mask=sampling.seen_token_mask.index_select(0, index),
        ),
    )


def test_talker_decode_executor_is_request_local_permutation_and_padding_safe(monkeypatch) -> None:
    def greedy(logits, sampling):
        tokens = logits.argmax(1)
        sampling.seen.scatter_(1, tokens[:, None], True)
        return tokens

    monkeypatch.setattr(TalkerCudaExecutor, "_sample", staticmethod(greedy))
    cache = _Cache()
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=cache,
        capture_slots=2,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(4)
    executor.capture(key)
    request_ids = ("a", "b", "c")
    batch = executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(request_ids=request_ids, rows=talker_decode_rows(_values((0, 1, 2)))),
    )
    singletons = [
        executor.replay(
            key,
            TalkerDecodeRowsExecutionInput(request_ids=(request_ids[row],), rows=talker_decode_rows(_values((row,)))),
        )
        for row in range(3)
    ]
    permutation = (2, 0, 1)
    permuted = executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(
            request_ids=tuple(request_ids[index] for index in permutation),
            rows=talker_decode_rows(_values(permutation)),
        ),
    )
    inverse = torch.argsort(torch.tensor(permutation))
    assert torch.equal(batch.result.tokens, torch.cat([result.result.tokens for result in singletons]))
    assert torch.equal(batch.result.last_hidden, torch.cat([result.result.last_hidden for result in singletons]))
    assert torch.equal(batch.result.tokens, permuted.result.tokens.index_select(0, inverse))
    assert torch.equal(batch.result.last_hidden, permuted.result.last_hidden.index_select(0, inverse))
    assert torch.equal(batch.next_sampling_offsets, torch.tensor([512, 544, 576]))
    assert [publication.request_id for publication in batch.kv_publications] == list(request_ids)
    static = talker_decode_static(executor, key)
    assert static.text_token_ids[3].item() == 0
    assert static.suppress_eos[3].item()
    assert static.sampling.temperature[3].item() == 0
    assert static.sampling.top_k[3].item() == 1
    assert static.sampling.top_p[3].item() == 1
    assert static.sampling.seed[3].item() == 0
    assert static.sampling.offsets[3].item() == 0


def test_talker_decode_rows_stage_directly_into_the_captured_slot(monkeypatch) -> None:
    def greedy(logits, sampling):
        tokens = logits.argmax(1)
        sampling.seen.scatter_(1, tokens[:, None], True)
        return tokens

    monkeypatch.setattr(TalkerCudaExecutor, "_sample", staticmethod(greedy))
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(4)
    executor.capture(key)
    source = _values((0, 1, 2))
    rows = tuple(
        TalkerDecodeExecutionRow(
            talker_step_embed=source.talker_step_embed[row],
            text_token_id=source.text_token_ids[row : row + 1],
            suppress_eos=bool(source.suppress_eos[row]),
            sampling=TalkerSamplingExecutionRow(
                temperature=float(source.sampling.temperature[row]),
                top_k=int(source.sampling.top_k[row]),
                top_p=float(source.sampling.top_p[row]),
                repetition_penalty=float(source.sampling.repetition_penalty[row]),
                seed=int(source.sampling.seed[row]),
                offset=int(source.sampling.offsets[row]),
                seen_token_mask=source.sampling.seen_token_mask[row],
            ),
        )
        for row in range(3)
    )

    direct = executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(request_ids=("a", "b", "c"), rows=rows),
    )
    singletons = [
        executor.replay(
            key,
            TalkerDecodeRowsExecutionInput(
                request_ids=(("a", "b", "c")[row],),
                rows=(rows[row],),
            ),
        )
        for row in range(3)
    ]
    permutation = (2, 0, 1)
    permuted = executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(
            request_ids=tuple(("a", "b", "c")[row] for row in permutation),
            rows=tuple(rows[row] for row in permutation),
        ),
    )
    inverse = torch.argsort(torch.tensor(permutation))

    static = talker_decode_static(executor, key)
    torch.testing.assert_close(
        static.talker_step_embed[:3],
        source.talker_step_embed.index_select(0, torch.tensor(permutation)),
    )
    assert static.text_token_ids[:3].tolist() == source.text_token_ids[list(permutation)].tolist()
    assert static.sampling.seed[:3].tolist() == [29, 3, 11]
    assert direct.next_sampling_offsets.tolist() == [512, 544, 576]
    assert torch.equal(
        direct.result.tokens,
        torch.cat([result.result.tokens for result in singletons]),
    )
    assert torch.equal(direct.result.tokens, permuted.result.tokens.index_select(0, inverse))


def test_talker_decode_forwards_the_typed_attention_plan_reuse_contract() -> None:
    cache = _Cache()
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=cache,
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(1)
    executor.capture(key)

    executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(
            request_ids=("a",),
            rows=talker_decode_rows(_values((0,), rows=1)),
            reuse_attention_plan=True,
        ),
    )

    assert cache.prepared[-1][-1] is True


@pytest.mark.parametrize("malformed", ["hidden", "seen"])
def test_talker_decode_direct_rows_reject_executor_shape_mismatch(malformed: str) -> None:
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(1)
    executor.capture(key)
    sampling = _direct_sampling_row()
    row = TalkerDecodeExecutionRow(
        talker_step_embed=torch.ones(3),
        text_token_id=torch.tensor(1, dtype=torch.long),
        suppress_eos=False,
        sampling=sampling,
    )
    if malformed == "hidden":
        row = replace(row, talker_step_embed=torch.ones(2))
    else:
        row = replace(
            row,
            sampling=replace(
                sampling,
                seen_token_mask=torch.zeros(3, dtype=torch.bool),
            ),
        )

    with pytest.raises(ValueError, match="hidden|vocabulary"):
        executor.replay(
            key,
            TalkerDecodeRowsExecutionInput(request_ids=("a",), rows=(row,)),
        )


def test_talker_capture_ingress_rejects_invalid_sampling_domain(monkeypatch) -> None:
    monkeypatch.setattr(
        TalkerCudaExecutor,
        "_sample",
        staticmethod(lambda logits, sampling: logits.argmax(1)),
    )
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(1)
    executor.capture(key)
    source = _values((0,), rows=1)
    invalid = replace(
        source,
        sampling=replace(source.sampling, top_p=torch.zeros(1)),
    )

    with pytest.raises(ValueError, match="top_p"):
        executor.replay(
            key,
            TalkerDecodeRowsExecutionInput(request_ids=("a",), rows=talker_decode_rows(invalid)),
        )


def test_talker_decode_b1_direct_rows_do_not_build_intermediate_aggregate_tensors(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        TalkerCudaExecutor,
        "_sample",
        staticmethod(lambda logits, sampling: logits.argmax(1)),
    )
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(1)
    executor.capture(key)
    source = _values((0,), rows=1)
    row = TalkerDecodeExecutionRow(
        talker_step_embed=source.talker_step_embed[0],
        text_token_id=source.text_token_ids[0:1],
        suppress_eos=bool(source.suppress_eos[0]),
        sampling=TalkerSamplingExecutionRow(
            temperature=float(source.sampling.temperature[0]),
            top_k=int(source.sampling.top_k[0]),
            top_p=float(source.sampling.top_p[0]),
            repetition_penalty=float(source.sampling.repetition_penalty[0]),
            seed=int(source.sampling.seed[0]),
            offset=int(source.sampling.offsets[0]),
            seen_token_mask=source.sampling.seen_token_mask[0],
        ),
    )
    static = talker_decode_static(executor, key)
    destinations = {
        static.talker_step_embed.data_ptr(),
        static.text_token_ids.data_ptr(),
    }
    aggregate_ops: list[str] = []
    original_stack = torch.stack
    original_cat = torch.cat

    def stack_spy(*args, **kwargs):
        output = kwargs.get("out")
        if output is not None and output.data_ptr() in destinations:
            aggregate_ops.append("stack")
        return original_stack(*args, **kwargs)

    def cat_spy(*args, **kwargs):
        output = kwargs.get("out")
        if output is not None and output.data_ptr() in destinations:
            aggregate_ops.append("cat")
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "stack", stack_spy)
    monkeypatch.setattr(torch, "cat", cat_spy)

    executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(request_ids=("a",), rows=(row,)),
    )

    assert aggregate_ops == []


def test_talker_prefill_rows_pack_directly_into_the_captured_slot(monkeypatch) -> None:
    monkeypatch.setattr(
        TalkerCudaExecutor,
        "_sample",
        staticmethod(lambda logits, sampling: logits.argmax(1)),
    )
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerPrefillCaptureKey(2, 4, None)
    executor.capture(key)

    sampling = lambda seed: TalkerSamplingExecutionRow(  # noqa: E731 - compact row fixture
        0.0,
        1,
        1.0,
        1.0,
        seed,
        0,
        None,
    )
    rows = (
        TalkerPrefillExecutionRow(
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
            torch.tensor([False, True]),
            True,
            sampling(7),
        ),
        TalkerPrefillExecutionRow(
            torch.tensor([5]),
            torch.tensor([6]),
            torch.tensor([True]),
            True,
            sampling(11),
        ),
    )

    result = executor.replay(
        key,
        TalkerPrefillRowsExecutionInput(request_ids=("a", "b"), rows=rows),
    )
    singletons = [
        executor.replay(
            key,
            TalkerPrefillRowsExecutionInput(
                request_ids=(("a", "b")[row],),
                rows=(rows[row],),
            ),
        )
        for row in range(2)
    ]
    permuted = executor.replay(
        key,
        TalkerPrefillRowsExecutionInput(
            request_ids=("b", "a"),
            rows=(rows[1], rows[0]),
        ),
    )

    assert result.result.tokens.shape == (2,)
    static = talker_prefill_static(executor, key)
    assert static.text_token_ids.tolist() == [5, 1, 2, 0]
    assert static.codec_token_ids.tolist() == [6, 3, 4, 0]
    assert static.last_token_indices.tolist() == [0, 2]
    assert static.sampling.seed.tolist() == [11, 7]
    assert torch.equal(
        result.result.tokens,
        torch.cat([item.result.tokens for item in singletons]),
    )
    assert torch.equal(result.result.tokens, permuted.result.tokens.flip(0))


def test_talker_prefill_capture_slot_owns_separate_host_token_staging() -> None:
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerPrefillCaptureKey(2, 4, None)

    executor.capture(key)

    slot = executor._prefill[key][0]
    assert slot.host_text.device.type == "cpu"
    assert slot.host_codec.device.type == "cpu"
    assert slot.host_mask.device.type == "cpu"
    assert slot.host_text.shape == slot.text.shape
    assert slot.host_codec.shape == slot.codec.shape
    assert slot.host_mask.shape == slot.mask.shape
    assert slot.host_text.data_ptr() != slot.text.data_ptr()
    assert slot.host_codec.data_ptr() != slot.codec.data_ptr()
    assert slot.host_mask.data_ptr() != slot.mask.data_ptr()


def test_talker_prefill_b1_packs_into_static_storage_without_torch_cat(monkeypatch) -> None:
    monkeypatch.setattr(
        TalkerCudaExecutor,
        "_sample",
        staticmethod(lambda logits, sampling: logits.argmax(1)),
    )
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerPrefillCaptureKey(1, 4, None)
    executor.capture(key)
    row = TalkerPrefillExecutionRow(
        text_token_ids=torch.tensor([1, 2]),
        codec_token_ids=torch.tensor([3, 4]),
        codec_token_mask=torch.tensor([False, True]),
        suppress_eos=True,
        sampling=TalkerSamplingExecutionRow(0.0, 1, 1.0, 1.0, 7, 0, None),
    )
    static = talker_prefill_static(executor, key)
    destinations = {
        static.text_token_ids.data_ptr(),
        static.codec_token_ids.data_ptr(),
        static.codec_token_mask.data_ptr(),
    }
    aggregate_outputs: list[int] = []
    original_cat = torch.cat

    def cat_spy(*args, **kwargs):
        output = kwargs.get("out")
        if output is not None and output.data_ptr() in destinations:
            aggregate_outputs.append(output.data_ptr())
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", cat_spy)

    executor.replay(
        key,
        TalkerPrefillRowsExecutionInput(request_ids=("a",), rows=(row,)),
    )

    assert aggregate_outputs == []


def test_talker_seen_masks_use_one_grouped_copy_for_active_direct_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        TalkerCudaExecutor,
        "_sample",
        staticmethod(lambda logits, sampling: logits.argmax(1)),
    )
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(3)
    executor.capture(key)
    source = _values((0, 1, 2))
    masks = (
        torch.tensor([True, False, False, False]),
        torch.tensor([False, True, False, False]),
        torch.tensor([False, False, True, False]),
    )
    rows = tuple(
        TalkerDecodeExecutionRow(
            talker_step_embed=source.talker_step_embed[index],
            text_token_id=source.text_token_ids[index : index + 1],
            suppress_eos=bool(source.suppress_eos[index]),
            sampling=TalkerSamplingExecutionRow(
                temperature=float(source.sampling.temperature[index]),
                top_k=int(source.sampling.top_k[index]),
                top_p=float(source.sampling.top_p[index]),
                repetition_penalty=float(source.sampling.repetition_penalty[index]),
                seed=int(source.sampling.seed[index]),
                offset=int(source.sampling.offsets[index]),
                seen_token_mask=masks[index],
            ),
        )
        for index in range(3)
    )
    calls: list[int] = []
    original_foreach_copy = torch._foreach_copy_

    def foreach_copy_spy(destinations, sources, *args, **kwargs):
        calls.append(len(destinations))
        return original_foreach_copy(destinations, sources, *args, **kwargs)

    monkeypatch.setattr(torch, "_foreach_copy_", foreach_copy_spy)

    executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(request_ids=("a", "b", "c"), rows=rows),
    )

    assert calls == [3]


def test_talker_executor_rotates_exclusive_slots() -> None:
    cache = _Cache()
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=cache,
        capture_slots=2,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(4)
    executor.capture(key)
    executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(request_ids=("a", "b", "c"), rows=talker_decode_rows(_values((0, 1, 2)))),
    )
    executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(request_ids=("a", "b", "c"), rows=talker_decode_rows(_values((0, 1, 2)))),
    )
    replay_contexts = [row[1] for row in cache.prepared if row[0] == "decode"]
    assert replay_contexts[0] is not replay_contexts[1]


def test_talker_replay_failure_aborts_pending_kv_and_releases_slot_for_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        TalkerCudaExecutor,
        "_sample",
        staticmethod(lambda logits, sampling: logits.argmax(1)),
    )
    cache = _AbortRecordingCache()
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=cache,
        capture_slots=1,
        driver=_Driver(),
    )
    key = TalkerDecodeCaptureKey(1)
    executor.capture(key)
    slot = executor._decode[key][0]
    slot.call = _FailOnceCaptured(slot.call)
    values = TalkerDecodeRowsExecutionInput(request_ids=("a",), rows=talker_decode_rows(_values((0,), rows=1)))

    with pytest.raises(RuntimeError, match="injected replay failure"):
        executor.replay(key, values)

    assert cache.aborted == [("a",)]
    retry = executor.replay(key, values)
    assert retry.result.tokens.shape == (1,)


def test_talker_captured_outputs_escape_static_storage_with_exactly_one_clone(monkeypatch) -> None:
    monkeypatch.setattr(
        TalkerCudaExecutor,
        "_sample",
        staticmethod(lambda logits, sampling: logits.argmax(1)),
    )
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_PersistentDriver(),
    )
    key = TalkerDecodeCaptureKey(1)
    executor.capture(key)
    slot = executor._decode[key][0]
    assert isinstance(slot.call, _PersistentCaptured)
    captured = slot.call.output
    assert isinstance(captured, tuple)
    static_pointers = {value.data_ptr() for value in captured}
    static_pointers.add(slot.sampling.seen.data_ptr())
    clone_counts = dict.fromkeys(static_pointers, 0)
    original_clone = torch.Tensor.clone

    def clone_spy(value, *args, **kwargs):
        pointer = value.data_ptr()
        if pointer in clone_counts:
            clone_counts[pointer] += 1
        return original_clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone_spy)

    result = executor.replay(
        key,
        TalkerDecodeRowsExecutionInput(request_ids=("a",), rows=talker_decode_rows(_values((0,), rows=1))),
    )

    assert set(clone_counts.values()) == {1}
    assert result.result.tokens.data_ptr() not in static_pointers
    assert result.result.last_hidden.data_ptr() not in static_pointers
    assert result.result.logits.data_ptr() not in static_pointers
    assert result.next_seen_token_masks.data_ptr() not in static_pointers


def test_talker_executor_respects_shared_cuda_graph_pool_ownership() -> None:
    fence = CudaGraphPoolFence(device=torch.device("cpu"))
    executor = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=_Cache(),
        capture_slots=1,
        driver=_Driver(),
        submission_fence=fence,
    )
    key = TalkerDecodeCaptureKey(4)
    executor.capture(key)
    lease = fence.reserve()
    try:
        with pytest.raises(SlotBusyError, match="CUDA Graph memory pool"):
            executor.replay(
                key,
                TalkerDecodeRowsExecutionInput(
                    request_ids=("a", "b", "c"),
                    rows=talker_decode_rows(_values((0, 1, 2))),
                ),
            )
    finally:
        fence.release(lease)

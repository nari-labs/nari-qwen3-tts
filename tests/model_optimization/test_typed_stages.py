from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from model_optimization.rows import codec_rows, codec_static, cp_rows, cp_static
from nari_qwen3_tts.contract import (
    CodecCaptureKey,
    CodecExecutionMode,
    CodePredictorCaptureKey,
)
from nari_qwen3_tts.contract.frames import WARM_TEMPLATE_FRAMES
from nari_qwen3_tts.contract.rng import CodePredictorSamplerRoute
from nari_qwen3_tts.executor import (
    CodecExecutionRow,
    CodecRowsExecutionInput,
    CodePredictorExecutionRow,
    CodePredictorRowsExecutionInput,
    CudaGraphPoolFence,
    SlotBusyError,
)
from nari_qwen3_tts.executor import (
    CodecExecutor as CodecCudaExecutor,
)
from nari_qwen3_tts.executor import (
    CodePredictorExecutor as CodePredictorCudaExecutor,
)
from nari_qwen3_tts.executor.types import CodePredictorInput
from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState

# The warm state template is decoded cold at WARM_TEMPLATE_FRAMES, so the
# executor requires that shape to be one it captures.
_COLD_FRAME_SIZES = (4, 5, 6, WARM_TEMPLATE_FRAMES)
_CODEC_STATE_MAPPINGS = (
    "transformer_keys",
    "transformer_values",
    "conv_histories",
    "transconv_overlaps",
)


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
        if isinstance(self.output, tuple):
            assert isinstance(fresh, tuple)
            for destination, source in zip(self.output, fresh, strict=True):
                destination.copy_(source)
        else:
            assert isinstance(self.output, torch.Tensor)
            assert isinstance(fresh, torch.Tensor)
            self.output.copy_(fresh)
        return self.output


class _PersistentDriver:
    def capture(self, operation, *, after_warmup=None):
        output = operation()
        if after_warmup is not None:
            after_warmup()
        return _PersistentCaptured(operation, output)


class _FailOnceCaptured:
    def __init__(self, successor):
        self.successor = successor
        self.failed = False

    def replay(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected replay failure")
        return self.successor.replay()


class _CodePredictorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.small_to_mtp_projection = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.small_to_mtp_projection.weight.copy_(torch.eye(4))
        embeddings = [torch.nn.Embedding(16, 4) for _ in range(3)]
        self.model = torch.nn.Module()
        self.model.codec_embedding = torch.nn.ModuleList(embeddings)
        self.lm_head_weight = torch.nn.Parameter(
            torch.arange(3 * 16 * 4, dtype=torch.float32).reshape(3, 16, 4) / 100,
            requires_grad=False,
        )

    @staticmethod
    def forward_depth_unrolled(
        inputs: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, kv_cache, kwargs
        return inputs


def _cp_config() -> object:
    return SimpleNamespace(
        num_code_groups=4,
        talker=SimpleNamespace(hidden_size=4),
        code_predictor=SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
        ),
    )


def _cp_layer0() -> torch.nn.Embedding:
    embedding = torch.nn.Embedding(16, 4)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4) / 50
        )
    return embedding


def _cp_values(order: tuple[int, ...]) -> CodePredictorInput:
    base_hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return CodePredictorInput(
        layer0_token=torch.tensor([3, 5, 7], dtype=torch.long)[list(order)],
        past_hidden=base_hidden[list(order)],
        temperature=torch.ones(3)[list(order)],
        top_k=torch.tensor([2, 3, 4], dtype=torch.int32)[list(order)],
        top_p=torch.tensor([0.7, 0.8, 0.9])[list(order)],
        seed=torch.tensor([11, 23, 37], dtype=torch.long)[list(order)],
        offsets=torch.tensor([[1, 2, 3], [5, 6, 7], [9, 10, 11]], dtype=torch.long)[list(order)],
        num_code_groups=4,
    )


def _direct_cp_row() -> CodePredictorExecutionRow:
    return CodePredictorExecutionRow(
        layer0_token=torch.tensor(3, dtype=torch.long),
        past_hidden=torch.ones(4),
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        seed=7,
        offsets=(1, 2, 3),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", float("nan")),
        ("top_k", -1),
        ("top_k", True),
        ("top_p", 0.0),
        ("seed", -1),
        ("offsets", (1, -1, 3)),
    ],
)
def test_direct_code_predictor_rows_reject_invalid_logical_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        replace(_direct_cp_row(), **{field: value})


def test_direct_code_predictor_rows_validate_tensor_contracts_before_staging() -> None:
    with pytest.raises(TypeError, match="layer0_token"):
        replace(_direct_cp_row(), layer0_token=torch.tensor(3.0))
    with pytest.raises(ValueError, match="past_hidden"):
        replace(_direct_cp_row(), past_hidden=torch.ones((1, 4)))


def test_direct_codec_rows_validate_frame_contract_before_staging() -> None:
    with pytest.raises(TypeError, match="Codec row frame"):
        CodecExecutionRow(
            frames=(torch.ones(2),),
            state=IncrementalCodecState(),
        )


def test_code_predictor_executor_is_singleton_batch_permutation_and_padding_safe() -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    assert executor.captured_cuda_graph_instances == 2
    values = _cp_values((0, 1, 2))
    batch = executor.replay(key, CodePredictorRowsExecutionInput(rows=cp_rows(values)))
    singletons = [
        executor.replay(
            key,
            CodePredictorRowsExecutionInput(
                rows=cp_rows(CodePredictorInput(
                    layer0_token=values.layer0_token[row : row + 1],
                    past_hidden=values.past_hidden[row : row + 1],
                    temperature=values.temperature[row : row + 1],
                    top_k=values.top_k[row : row + 1],
                    top_p=values.top_p[row : row + 1],
                    seed=values.seed[row : row + 1],
                    offsets=values.offsets[row : row + 1],
                    num_code_groups=4,
                )),
            ),
        )
        for row in range(3)
    ]
    permutation = (2, 0, 1)
    permuted = executor.replay(key, CodePredictorRowsExecutionInput(rows=cp_rows(_cp_values(permutation))))
    inverse = torch.argsort(torch.tensor(permutation))
    assert torch.equal(batch.frames, torch.cat([result.frames for result in singletons]))
    assert torch.equal(batch.codec_sum, torch.cat([result.codec_sum for result in singletons]))
    assert torch.equal(batch.frames, permuted.frames.index_select(0, inverse))
    assert torch.equal(batch.codec_sum, permuted.codec_sum.index_select(0, inverse))
    slot = cp_static(executor, key)
    assert slot.layer0_token[3].item() == 0
    assert slot.seed[3].item() == 0
    assert torch.count_nonzero(slot.offsets[3]) == 0


def test_code_predictor_rows_stage_directly_without_an_aggregate_input_tensor() -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    source = _cp_values((0, 1, 2))
    rows = tuple(
        CodePredictorExecutionRow(
            layer0_token=source.layer0_token[row],
            past_hidden=source.past_hidden[row],
            temperature=float(source.temperature[row]),
            top_k=int(source.top_k[row]),
            top_p=float(source.top_p[row]),
            seed=int(source.seed[row]),
            offsets=tuple(int(value) for value in source.offsets[row]),
        )
        for row in range(3)
    )

    direct = executor.replay(
        key,
        CodePredictorRowsExecutionInput(rows=rows),
    )
    singletons = [
        executor.replay(
            key,
            CodePredictorRowsExecutionInput(rows=(rows[row],)),
        )
        for row in range(3)
    ]
    permutation = (2, 0, 1)
    permuted = executor.replay(
        key,
        CodePredictorRowsExecutionInput(
            rows=tuple(rows[row] for row in permutation),
        ),
    )
    inverse = torch.argsort(torch.tensor(permutation))

    static = cp_static(executor, key)
    assert static.layer0_token[:3].tolist() == [7, 3, 5]
    torch.testing.assert_close(static.past_hidden[:3], source.past_hidden[list(permutation)])
    assert static.offsets[:3].tolist() == source.offsets[list(permutation)].tolist()
    assert direct.frames.shape == (3, 4)
    assert torch.equal(direct.frames, torch.cat([result.frames for result in singletons]))
    assert torch.equal(direct.frames, permuted.frames.index_select(0, inverse))
    assert torch.equal(direct.codec_sum, permuted.codec_sum.index_select(0, inverse))


def test_code_predictor_direct_rows_reject_executor_hidden_shape_mismatch() -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(1)
    executor.capture(key)

    with pytest.raises(ValueError, match="past_hidden"):
        executor.replay(
            key,
            CodePredictorRowsExecutionInput(
                rows=(replace(_direct_cp_row(), past_hidden=torch.ones(3)),),
                ),
        )


def test_code_predictor_b1_direct_row_does_not_stack_an_aggregate_input(monkeypatch) -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(1)
    executor.capture(key)
    source = _cp_values((0,))
    row = CodePredictorExecutionRow(
        layer0_token=source.layer0_token[0],
        past_hidden=source.past_hidden[0],
        temperature=float(source.temperature[0]),
        top_k=int(source.top_k[0]),
        top_p=float(source.top_p[0]),
        seed=int(source.seed[0]),
        offsets=tuple(int(value) for value in source.offsets[0]),
    )
    static = cp_static(executor, key)
    destinations = {static.layer0_token.data_ptr(), static.past_hidden.data_ptr()}
    aggregate_outputs: list[int] = []
    original_stack = torch.stack

    def stack_spy(*args, **kwargs):
        output = kwargs.get("out")
        if output is not None and output.data_ptr() in destinations:
            aggregate_outputs.append(output.data_ptr())
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(torch, "stack", stack_spy)

    executor.replay(
        key,
        CodePredictorRowsExecutionInput(rows=(row,)),
    )

    assert aggregate_outputs == []


def test_code_predictor_host_metadata_is_packed_without_tensor_scalar_writes(
    monkeypatch,
) -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    slot = executor._slots[key]
    host_pointers = {
        slot.host_temperature.data_ptr(),
        slot.host_top_k.data_ptr(),
        slot.host_top_p.data_ptr(),
        slot.host_seed.data_ptr(),
        slot.host_offsets.data_ptr(),
    }
    scalar_writes: list[object] = []
    original_setitem = torch.Tensor.__setitem__

    def setitem_spy(tensor, key, value):
        if tensor.data_ptr() in host_pointers:
            scalar_writes.append(key)
        return original_setitem(tensor, key, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", setitem_spy)

    executor.replay(
        key,
        CodePredictorRowsExecutionInput(rows=cp_rows(_cp_values((0, 1, 2)))),
    )

    assert scalar_writes == []


def test_code_predictor_direct_rows_stage_request_specific_position_ids() -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(1)
    executor.capture(key)
    source = _cp_values((0,))
    expected_positions = torch.tensor([7, 8, 9, 10])
    row = CodePredictorExecutionRow(
        layer0_token=source.layer0_token[0],
        past_hidden=source.past_hidden[0],
        temperature=float(source.temperature[0]),
        top_k=int(source.top_k[0]),
        top_p=float(source.top_p[0]),
        seed=int(source.seed[0]),
        offsets=tuple(int(value) for value in source.offsets[0]),
        position_ids=expected_positions,
    )

    executor.replay(
        key,
        CodePredictorRowsExecutionInput(rows=(row,)),
    )

    assert cp_static(executor, key).position_ids is not None
    assert torch.equal(cp_static(executor, key).position_ids[0], expected_positions)


def test_code_predictor_outputs_escape_static_capture_storage_once(monkeypatch) -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_PersistentDriver(),
    )
    key = CodePredictorCaptureKey(1)
    executor.capture(key)
    slot = executor._slots[key]
    assert isinstance(slot.fused_call, _PersistentCaptured)
    captured = slot.fused_call.output
    assert isinstance(captured, tuple)
    static_pointers = {value.data_ptr() for value in captured}
    clone_counts = dict.fromkeys(static_pointers, 0)
    original_clone = torch.Tensor.clone

    def clone_spy(value, *args, **kwargs):
        pointer = value.data_ptr()
        if pointer in clone_counts:
            clone_counts[pointer] += 1
        return original_clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone_spy)

    # Production replay runs under the Engine thread's inference-mode scope.
    # The persistent fake driver must exercise the same tensor mutation rules.
    with torch.inference_mode():
        result = executor.replay(
            key,
            CodePredictorRowsExecutionInput(rows=cp_rows(_cp_values((0,)))),
        )

    assert set(clone_counts.values()) == {1}
    assert result.frames.data_ptr() not in static_pointers
    assert result.codec_sum.data_ptr() not in static_pointers


@pytest.mark.parametrize("general_top_k", [0, 65, 4096])
def test_code_predictor_rejects_mixed_sampler_routes_before_replay(
    general_top_k: int,
) -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    base = _cp_values((0, 1, 2))
    values = CodePredictorInput(
        layer0_token=base.layer0_token,
        past_hidden=base.past_hidden,
        temperature=base.temperature,
        top_k=torch.tensor([general_top_k, 1, 1], dtype=torch.int32),
        top_p=base.top_p,
        seed=base.seed,
        offsets=base.offsets,
        num_code_groups=base.num_code_groups,
    )

    with pytest.raises(ValueError, match="homogeneous sampler route"):
        executor.replay(
            key,
            CodePredictorRowsExecutionInput(
                rows=cp_rows(values),
                sampler_route=CodePredictorSamplerRoute.GENERAL,
            ),
        )


@pytest.mark.parametrize("general_top_k", [0, 65, 4096])
def test_code_predictor_admits_homogeneous_general_sampler_rows(
    general_top_k: int,
) -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    base = _cp_values((0, 1, 2))
    values = replace(
        base,
        top_k=torch.full((3,), general_top_k, dtype=torch.int32),
    )

    result = executor.replay(
        key,
        CodePredictorRowsExecutionInput(
            rows=cp_rows(values),
            sampler_route=CodePredictorSamplerRoute.GENERAL,
        ),
    )

    assert result.frames.shape == (3, 4)


def test_code_predictor_admits_greedy_rows_at_any_top_k() -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    base = _cp_values((0, 1, 2))
    values = CodePredictorInput(
        layer0_token=base.layer0_token,
        past_hidden=base.past_hidden,
        temperature=torch.zeros(3),
        top_k=torch.tensor([0, 65, 4096], dtype=torch.int32),
        top_p=base.top_p,
        seed=base.seed,
        offsets=base.offsets,
        num_code_groups=base.num_code_groups,
    )

    result = executor.replay(key, CodePredictorRowsExecutionInput(rows=cp_rows(values)))

    assert result.frames.shape == (3, 4)
    assert cp_static(executor, key).top_k[:3].tolist() == [0, 65, 4096]


def test_code_predictor_direct_rows_stage_without_any_device_scalar_read(
    monkeypatch,
) -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    slot = executor._slots[key]
    captured = slot.fused_call.replay()
    slot.fused_call = _Captured(lambda: captured)
    monkeypatch.setattr(
        torch,
        "any",
        lambda value: (_ for _ in ()).throw(AssertionError("unexpected scalar read")),
    )

    source = _cp_values((0, 1, 2))
    rows = tuple(
        CodePredictorExecutionRow(
            layer0_token=source.layer0_token[row],
            past_hidden=source.past_hidden[row],
            temperature=float(source.temperature[row]),
            top_k=int(source.top_k[row]),
            top_p=float(source.top_p[row]),
            seed=int(source.seed[row]),
            offsets=tuple(int(value) for value in source.offsets[row]),
        )
        for row in range(3)
    )
    result = executor.replay(
        key,
        CodePredictorRowsExecutionInput(rows=rows),
    )

    assert result.frames.shape == (3, 4)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("temperature", torch.tensor([float("nan"), 1.0, 1.0])),
        ("top_k", torch.tensor([-1, 1, 1], dtype=torch.int32)),
        ("top_p", torch.tensor([0.0, 1.0, 1.0])),
        ("seed", torch.tensor([-1, 1, 1], dtype=torch.long)),
        ("offsets", torch.tensor([[-1, 2, 3], [5, 6, 7], [9, 10, 11]], dtype=torch.long)),
    ],
)
def test_code_predictor_capture_ingress_rejects_invalid_sampling_domain(
    field: str,
    replacement: torch.Tensor,
) -> None:
    executor = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    key = CodePredictorCaptureKey(4)
    executor.capture(key)
    source = _cp_values((0, 1, 2))
    values = CodePredictorInput(
        **{
            name: replacement if name == field else getattr(source, name)
            for name in (
                "layer0_token",
                "past_hidden",
                "temperature",
                "top_k",
                "top_p",
                "seed",
                "offsets",
                "num_code_groups",
                "position_ids",
            )
        }
    )

    with pytest.raises(ValueError, match=field):
        executor.replay(
            key,
            CodePredictorRowsExecutionInput(rows=cp_rows(values)),
        )


class _Incremental:
    samples_per_frame = 2
    retained_context = 8

    def __call__(self, codes, states, **kwargs):
        del kwargs
        frames = codes.shape[2]
        for row, state in enumerate(states):
            state.frame_position += frames
            state.transformer_context_length += frames
            state.conv_histories["x"] = codes[row, :1, -1:].float().clone()
        return codes.float().sum(1, keepdim=True).repeat_interleave(2, dim=2)


class _Codec:
    class _Decoder(torch.nn.Module):
        @staticmethod
        def forward(codes):
            return codes.float().sum(1, keepdim=True).repeat_interleave(2, dim=2)

    decoder = _Decoder()
    incremental_decoder = _Incremental()


class _RichIncremental:
    samples_per_frame = 2
    retained_context = 8

    def __call__(self, codes, states, **kwargs):
        del kwargs
        frames = codes.shape[2]
        for row, state in enumerate(states):
            marker = codes[row, :1, -1:].float()
            key_value = state.transformer_keys.get(0, marker.new_zeros((1, 2)))
            value_value = state.transformer_values.get(0, marker.new_zeros((1, 2)))
            conv_value = state.conv_histories.get("conv", marker.new_zeros((1, 1)))
            overlap_value = state.transconv_overlaps.get("up", marker.new_zeros((1, 1)))
            state.transformer_keys[0] = key_value + marker.expand_as(key_value)
            state.transformer_values[0] = value_value + marker.expand_as(value_value)
            state.conv_histories["conv"] = conv_value + marker
            state.transconv_overlaps["up"] = overlap_value + marker
            state.frame_position += frames
            state.transformer_context_length += frames
        bias = torch.stack(
            tuple(state.transformer_keys[0].reshape(-1)[0] for state in states),
        )[:, None, None]
        return (codes.float().sum(1, keepdim=True) + bias).repeat_interleave(2, dim=2)


class _RichCodec:
    decoder = _Codec._Decoder()
    incremental_decoder = _RichIncremental()


def _rich_warm_state(marker: int) -> IncrementalCodecState:
    state = IncrementalCodecState()
    codes = torch.full((1, 2, 7), marker, dtype=torch.long)
    _RichCodec.incremental_decoder(
        codes,
        [state],
        position_ids=torch.arange(7).unsqueeze(0),
        context_lengths=torch.zeros(1, dtype=torch.long),
    )
    return state


def test_codec_executor_keeps_input_state_pending_and_trims_terminal_padding() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 2)
    executor.capture(key)
    source = (IncrementalCodecState(),)
    result = executor.replay(
        key,
        CodecRowsExecutionInput(
            rows=codec_rows(torch.ones((1, 3, 2), dtype=torch.long), source),
            visible_frames=3,
            terminal=True,
        ),
    )
    assert result.pcm.shape == (1, 6)
    assert result.terminal
    assert source[0].frame_position == 0
    assert result.states is not None and result.states[0].frame_position == 4


def test_warm_codec_executor_batches_variable_terminal_tails_by_pad_shape() -> None:
    executor = CodecCudaExecutor(
        model=_RichCodec(),
        incremental_decoder=_RichCodec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.WARM, 4, 2)
    executor.capture(key)
    short = tuple(torch.full((2,), 1, dtype=torch.long) for _ in range(2))
    longer = tuple(torch.full((2,), 2, dtype=torch.long) for _ in range(3))

    result = executor.replay(
        key,
        CodecRowsExecutionInput(
            rows=(
                CodecExecutionRow(
                    frames=short,
                    state=_rich_warm_state(1),
                    visible_frames=2,
                ),
                CodecExecutionRow(
                    frames=longer,
                    state=_rich_warm_state(2),
                    visible_frames=3,
                ),
            ),
            visible_frames=0,
            terminal=True,
        ),
    )

    assert result.pcm.shape == (2, 6)
    assert result.pcm_lengths == (4, 6)
    assert result.pcm_row(0).shape == (4,)
    assert result.pcm_row(1).shape == (6,)
    assert torch.count_nonzero(result.pcm_row(0)) > 0
    assert torch.count_nonzero(result.pcm_row(1)) > 0


def test_whole_sequence_codec_executor_batches_row_local_pcm_windows_by_model_shape() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.WHOLE_SEQUENCE, 3, 2)
    executor.capture(key)
    frames = torch.arange(12, dtype=torch.long).reshape(2, 3, 2)
    rows = (
        CodecExecutionRow(
            frames=tuple(frames[0]),
            state=None,
            visible_frames=2,
            pcm_start_frame=1,
        ),
        CodecExecutionRow(
            frames=tuple(frames[1]),
            state=None,
            visible_frames=3,
            pcm_start_frame=0,
        ),
    )

    result = executor.replay(
        key,
        CodecRowsExecutionInput(rows=rows, visible_frames=0),
    )

    expected = CodecCudaExecutor._pcm16(
        frames.sum(2).repeat_interleave(2, 1).to(torch.float32)
    )
    assert result.pcm_lengths == (4, 6)
    assert torch.equal(result.pcm_row(0), expected[0, 2:])
    assert torch.equal(result.pcm_row(1), expected[1])


def test_codec_rejects_underfilled_nonterminal_capture() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 1)
    executor.capture(key)

    with pytest.raises(ValueError, match="exact.*frames|terminal padding"):
        executor.replay(
            key,
            CodecRowsExecutionInput(
                rows=codec_rows(torch.ones((1, 3, 2), dtype=torch.long), (IncrementalCodecState(),)),
                visible_frames=3,
                terminal=False,
            ),
        )


def test_codec_rows_stage_committed_frames_directly_into_the_captured_slot() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 2)
    executor.capture(key)
    frames = torch.arange(16, dtype=torch.long).reshape(2, 4, 2)
    rows = tuple(
        CodecExecutionRow(
            frames=tuple(frames[row, frame] for frame in range(4)),
            state=IncrementalCodecState(),
        )
        for row in range(2)
    )

    result = executor.replay(
        key,
        CodecRowsExecutionInput(
            rows=rows,
            visible_frames=4,
            terminal=False,
        ),
    )
    singletons = [
        executor.replay(
            key,
            CodecRowsExecutionInput(
                rows=(rows[row],),
                visible_frames=4,
            ),
        )
        for row in range(2)
    ]
    permutation = (1, 0)
    permuted = executor.replay(
        key,
        CodecRowsExecutionInput(
            rows=tuple(rows[row] for row in permutation),
            visible_frames=4,
        ),
    )

    assert result.pcm.shape == (2, 8)
    assert torch.equal(codec_static(executor, key), frames[list(permutation)])
    assert torch.equal(result.pcm, torch.cat([item.pcm for item in singletons]))
    assert torch.equal(result.pcm, permuted.pcm.index_select(0, torch.tensor(permutation)))


def test_codec_direct_rows_reject_underfilled_nonterminal_capture() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 1)
    executor.capture(key)
    row = CodecExecutionRow(
        frames=tuple(torch.ones(2, dtype=torch.long) for _ in range(3)),
        state=IncrementalCodecState(),
    )

    with pytest.raises(ValueError, match="exact.*frames|terminal padding"):
        executor.replay(
            key,
            CodecRowsExecutionInput(
                rows=(row,),
                visible_frames=3,
                terminal=False,
            ),
        )


def test_codec_b1_direct_row_does_not_stack_an_aggregate_frame_tensor(monkeypatch) -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 1)
    executor.capture(key)
    frames = torch.arange(8, dtype=torch.long).reshape(4, 2)
    row = CodecExecutionRow(
        frames=tuple(frames[index] for index in range(4)),
        state=IncrementalCodecState(),
    )
    destination = codec_static(executor, key)
    aggregate_outputs: list[int] = []
    original_stack = torch.stack

    def stack_spy(*args, **kwargs):
        output = kwargs.get("out")
        if output is not None and output.data_ptr() == destination.data_ptr():
            aggregate_outputs.append(output.data_ptr())
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(torch, "stack", stack_spy)

    executor.replay(
        key,
        CodecRowsExecutionInput(rows=(row,), visible_frames=4),
    )

    assert aggregate_outputs == []


def test_codec_executor_is_singleton_batch_permutation_padding_and_poison_safe() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 4)
    executor.capture(key)
    frames = torch.arange(3 * 4 * 2, dtype=torch.long).reshape(3, 4, 2)

    def replay(indices: tuple[int, ...]):
        index = torch.tensor(indices, dtype=torch.long)
        return executor.replay(
            key,
            CodecRowsExecutionInput(
                rows=codec_rows(frames.index_select(0, index), tuple(IncrementalCodecState() for _ in indices)),
                visible_frames=4,
            ),
        )

    batch = replay((0, 1, 2))
    singletons = [replay((row,)) for row in range(3)]
    permutation = (2, 0, 1)
    permuted = replay(permutation)
    inverse = torch.argsort(torch.tensor(permutation))
    assert torch.equal(batch.pcm, torch.cat([result.pcm for result in singletons]))
    assert torch.equal(batch.pcm, permuted.pcm.index_select(0, inverse))
    assert batch.states is not None and permuted.states is not None
    assert [state.frame_position for state in batch.states] == [4, 4, 4]
    assert all(
        torch.equal(
            batch.states[row].conv_histories["x"],
            permuted.states[int(inverse[row])].conv_histories["x"],
        )
        for row in range(3)
    )
    static = executor._slots[key]
    assert torch.count_nonzero(static.frames[3]) == 0


def test_codec_executor_requires_its_warm_template_shape_to_be_captured() -> None:
    """The warm state template is produced by a cold decode of that shape.

    Templating from a shape the catalog does not capture would size every warm
    capture from state no request can actually be in.
    """

    with pytest.raises(ValueError, match="captured cold shapes"):
        CodecCudaExecutor(
            model=_Codec(),
            incremental_decoder=_Codec.incremental_decoder,
            num_code_groups=2,
            cold_frame_sizes=tuple(
                size for size in _COLD_FRAME_SIZES if size != WARM_TEMPLATE_FRAMES
            ),
            device=torch.device("cpu"),
            driver=_Driver(),
        )


def test_codec_executor_rejects_lifecycle_and_shape_drift() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    cold = CodecCaptureKey(CodecExecutionMode.COLD, 4, 1)
    executor.capture(cold)
    stale = IncrementalCodecState(frame_position=1, transformer_context_length=1)
    stale.conv_histories["x"] = torch.ones(1, 1)
    with pytest.raises(ValueError, match="cold"):
        executor.replay(
            cold,
            CodecRowsExecutionInput(
                rows=codec_rows(torch.ones((1, 4, 2), dtype=torch.long), (stale,)),
                visible_frames=4,
            ),
        )
    with pytest.raises(ValueError, match="captured shape"):
        executor.replay(
            cold,
            CodecRowsExecutionInput(
                rows=codec_rows(torch.ones((1, 5, 2), dtype=torch.long), (IncrementalCodecState(),)),
                visible_frames=5,
            ),
        )


def _warm_executor() -> tuple[CodecCudaExecutor, CodecCaptureKey]:
    executor = CodecCudaExecutor(
        model=_RichCodec(),
        incremental_decoder=_RichCodec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.WARM, 2, 1)
    executor.capture(key)
    return executor, key


def test_warm_codec_state_schema_rejects_extra_mapping_keys() -> None:
    executor, key = _warm_executor()
    frames = torch.ones((1, 2, 2), dtype=torch.long)

    extra = _rich_warm_state(1)
    extra.conv_histories["unexpected"] = torch.ones((1, 1))
    with pytest.raises(ValueError, match="schema|mapping keys"):
        executor.replay(
            key,
            CodecRowsExecutionInput(rows=codec_rows(frames, (extra,)), visible_frames=2),
        )


def test_warm_codec_state_schema_rejects_silent_dtype_casts() -> None:
    executor, key = _warm_executor()
    frames = torch.ones((1, 2, 2), dtype=torch.long)
    wrong_dtype = _rich_warm_state(1)
    wrong_dtype.transformer_keys[0] = wrong_dtype.transformer_keys[0].to(torch.int64)
    with pytest.raises(ValueError, match="dtype"):
        executor.replay(
            key,
            CodecRowsExecutionInput(rows=codec_rows(frames, (wrong_dtype,)), visible_frames=2),
        )


def test_warm_codec_is_singleton_batch_permutation_and_input_state_immutable() -> None:
    executor = CodecCudaExecutor(
        model=_RichCodec(),
        incremental_decoder=_RichCodec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.WARM, 2, 2)
    executor.capture(key)
    frames = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=torch.long,
    )
    source = (_rich_warm_state(1), _rich_warm_state(3))
    before = tuple(state.clone() for state in source[0].transformer_keys.values())

    batch = executor.replay(
        key,
        CodecRowsExecutionInput(rows=codec_rows(frames, source), visible_frames=2),
    )
    singletons = tuple(
        executor.replay(
            key,
            CodecRowsExecutionInput(
                rows=codec_rows(frames[row : row + 1], (_rich_warm_state((1, 3)[row]),)),
                visible_frames=2,
            ),
        )
        for row in range(2)
    )
    permuted = executor.replay(
        key,
        CodecRowsExecutionInput(
            rows=codec_rows(frames.flip(0), (_rich_warm_state(3), _rich_warm_state(1))),
            visible_frames=2,
        ),
    )

    assert torch.equal(batch.pcm, torch.cat(tuple(item.pcm for item in singletons)))
    assert torch.equal(batch.pcm, permuted.pcm.flip(0))
    assert source[0].frame_position == 7
    assert all(
        torch.equal(before_value, after_value)
        for before_value, after_value in zip(
            before,
            source[0].transformer_keys.values(),
            strict=True,
        )
    )
    assert batch.states is not None
    assert [state.frame_position for state in batch.states] == [9, 9]


def test_warm_codec_state_ingress_and_owned_egress_use_grouped_copy(monkeypatch) -> None:
    executor = CodecCudaExecutor(
        model=_RichCodec(),
        incremental_decoder=_RichCodec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.WARM, 2, 2)
    executor.capture(key)
    slot = executor._slots[key]
    assert slot.state is not None
    # Position starts and retained-context lengths are staged together. A
    # batch must not submit one CUDA scalar write/add per request row.
    assert slot.state.metadata.shape == (2, key.capture_batch_size)
    assert slot.state.host_metadata.shape == slot.state.metadata.shape
    assert slot.state.host_metadata.device.type == "cpu"
    metadata_destination = slot.state.metadata.data_ptr()
    state_destinations = {
        entry.values[row].data_ptr()
        for entry in slot.state.inputs
        for row in range(key.capture_batch_size)
    }
    direct_state_copies: list[int] = []
    metadata_copies: list[int] = []
    clone_calls: list[int] = []
    empty_like_calls: list[int] = []
    foreach_groups: list[int] = []
    original_copy = torch.Tensor.copy_
    original_clone = torch.Tensor.clone
    original_empty_like = torch.empty_like
    original_foreach_copy = torch._foreach_copy_

    def copy_spy(value, source, *args, **kwargs):
        if value.data_ptr() in state_destinations:
            direct_state_copies.append(value.data_ptr())
        if value.data_ptr() == metadata_destination:
            metadata_copies.append(value.data_ptr())
        return original_copy(value, source, *args, **kwargs)

    def clone_spy(value, *args, **kwargs):
        clone_calls.append(value.data_ptr())
        return original_clone(value, *args, **kwargs)

    def empty_like_spy(value, *args, **kwargs):
        empty_like_calls.append(value.data_ptr())
        return original_empty_like(value, *args, **kwargs)

    def foreach_copy_spy(destinations, sources, *args, **kwargs):
        foreach_groups.append(len(destinations))
        return original_foreach_copy(destinations, sources, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_spy)
    monkeypatch.setattr(torch.Tensor, "clone", clone_spy)
    monkeypatch.setattr(torch, "empty_like", empty_like_spy)
    monkeypatch.setattr(torch, "_foreach_copy_", foreach_copy_spy)
    source = (_rich_warm_state(1), _rich_warm_state(3))
    result = executor.replay(
        key,
        CodecRowsExecutionInput(
            rows=codec_rows(torch.ones((2, 2, 2), dtype=torch.long), source),
            visible_frames=2,
        ),
    )

    assert result.states is not None
    assert direct_state_copies == []
    assert metadata_copies == [metadata_destination]
    # The one remaining clone owns escaped PCM; Codec state uses grouped
    # copies into one packed successor allocation per device/dtype instead of
    # one allocation and one kernel per tensor.
    assert len(clone_calls) == 1
    assert empty_like_calls == []
    assert foreach_groups == [8, 8]
    state_tensors = [
        value
        for state in result.states
        for mapping_name in _CODEC_STATE_MAPPINGS
        for value in getattr(state, mapping_name).values()
    ]
    assert len({value.untyped_storage().data_ptr() for value in state_tensors}) == 1


def test_codec_outputs_escape_static_capture_and_state_storage() -> None:
    executor = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_PersistentDriver(),
    )
    key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 1)
    executor.capture(key)
    slot = executor._slots[key]
    assert isinstance(slot.call, _PersistentCaptured)

    result = executor.replay(
        key,
        CodecRowsExecutionInput(
            rows=codec_rows(torch.ones((1, 4, 2), dtype=torch.long), (IncrementalCodecState(),)),
            visible_frames=4,
        ),
    )

    assert isinstance(slot.call.output, torch.Tensor)
    assert result.pcm.data_ptr() != slot.call.output.data_ptr()
    assert result.states is not None and slot.state is not None
    captured_history = slot.state.outputs[0].conv_histories["x"]
    escaped_history = result.states[0].conv_histories["x"]
    assert escaped_history.data_ptr() != captured_history.data_ptr()
    expected_history = escaped_history.clone()
    captured_history.fill_(99)
    assert torch.equal(escaped_history, expected_history)


def test_code_predictor_and_codec_release_owned_slots_after_replay_failure() -> None:
    cp = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
    )
    cp_key = CodePredictorCaptureKey(1)
    cp.capture(cp_key)
    cp_slot = cp._slots[cp_key]
    cp_slot.fused_call = _FailOnceCaptured(cp_slot.fused_call)
    cp_values = CodePredictorRowsExecutionInput(rows=cp_rows(_cp_values((0,))))
    with pytest.raises(RuntimeError, match="injected replay failure"):
        cp.replay(cp_key, cp_values)
    assert cp.replay(cp_key, cp_values).frames.shape == (1, 4)

    codec = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
    )
    codec_key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 1)
    codec.capture(codec_key)
    codec_slot = codec._slots[codec_key]
    codec_slot.call = _FailOnceCaptured(codec_slot.call)
    codec_values = CodecRowsExecutionInput(
        rows=codec_rows(
            torch.ones((1, 4, 2), dtype=torch.long),
            (IncrementalCodecState(),),
        ),
        visible_frames=4,
    )
    with pytest.raises(RuntimeError, match="injected replay failure"):
        codec.replay(codec_key, codec_values)
    assert codec.replay(codec_key, codec_values).pcm.shape == (1, 8)


def test_code_predictor_and_codec_can_share_a_fail_closed_submission_fence() -> None:
    fence = CudaGraphPoolFence(device=torch.device("cpu"))
    cp = CodePredictorCudaExecutor(
        model=_CodePredictorModel(),
        layer0_embedding=_cp_layer0(),
        config=_cp_config(),
        max_batch_size=4,
        driver=_Driver(),
        submission_fence=fence,
    )
    codec = CodecCudaExecutor(
        model=_Codec(),
        incremental_decoder=_Codec.incremental_decoder,
        num_code_groups=2,
        cold_frame_sizes=_COLD_FRAME_SIZES,
        device=torch.device("cpu"),
        driver=_Driver(),
        submission_fence=fence,
    )
    cp_key = CodePredictorCaptureKey(4)
    codec_key = CodecCaptureKey(CodecExecutionMode.COLD, 4, 1)
    cp.capture(cp_key)
    codec.capture(codec_key)
    lease = fence.reserve()
    try:
        with pytest.raises(SlotBusyError, match="CUDA Graph memory pool"):
            cp.replay(cp_key, CodePredictorRowsExecutionInput(rows=cp_rows(_cp_values((0, 1, 2)))))
        with pytest.raises(SlotBusyError, match="CUDA Graph memory pool"):
            codec.replay(
                codec_key,
                CodecRowsExecutionInput(
                    rows=codec_rows(torch.ones((1, 4, 2), dtype=torch.long), (IncrementalCodecState(),)),
                    visible_frames=4,
                ),
            )
    finally:
        fence.release(lease)

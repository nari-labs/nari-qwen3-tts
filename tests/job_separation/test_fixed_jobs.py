from __future__ import annotations

import pytest
import torch

from nari_qwen3_tts.executor.code_predictor import CodePredictorExecutor as CodePredictorCudaExecutor
from nari_qwen3_tts.executor.talker import TalkerExecutor as TalkerCudaExecutor
from nari_qwen3_tts.executor.types import (
    CodePredictorInput,
    TalkerDecodeInput,
    TalkerPrefillInput,
    TalkerSamplingInput,
)
from nari_qwen3_tts.model.components import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from nari_qwen3_tts.model.config import Qwen3TTSCodePredictorConfig, Qwen3TTSConfig, Qwen3TTSTalkerConfig


def _config() -> Qwen3TTSConfig:
    predictor = Qwen3TTSCodePredictorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=12,
        num_code_groups=4,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )
    talker = Qwen3TTSTalkerConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=20,
        text_vocab_size=32,
        text_hidden_size=12,
        num_code_groups=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        codec_bos_id=2,
        codec_eos_token_id=11,
        codec_think_id=3,
        codec_nothink_id=4,
        codec_pad_id=5,
        codec_think_bos_id=6,
        codec_think_eos_id=7,
        codec_language_id={"english": 8},
        spk_id={"aiden": 9},
        spk_is_dialect={"aiden": False},
        code_predictor=predictor,
    )
    return Qwen3TTSConfig(
        talker=talker,
        tts_bos_token_id=29,
        tts_eos_token_id=30,
        tts_pad_token_id=31,
        tokenizer_type="test",
    )


class _AttentionContext:
    def __init__(self) -> None:
        self.layers: list[int] = []
        self.advances = 0

    def select_layer(self, index: int) -> None:
        self.layers.append(index)

    def attend(
        self,
        attention: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del attention, key, value
        return torch.zeros_like(query)

    def advance_sequence_lengths(self) -> None:
        self.advances += 1


def _sampling(rows: int, vocab_size: int) -> TalkerSamplingInput:
    return TalkerSamplingInput(
        temperature=torch.zeros(rows),
        top_k=torch.zeros(rows, dtype=torch.int32),
        top_p=torch.ones(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.arange(rows, dtype=torch.long),
        offsets=torch.zeros(rows, dtype=torch.long),
        seen_token_mask=torch.zeros((rows, vocab_size), dtype=torch.bool),
    )


def _talker_executor(model: torch.nn.Module, config: Qwen3TTSConfig) -> TalkerCudaExecutor:
    return TalkerCudaExecutor(
        model=model,
        config=config,
        cache=object(),
        capture_slots=1,
        driver=object(),
    )


def _code_predictor_executor(
    model: torch.nn.Module,
    layer0_embedding: torch.nn.Embedding,
    config: Qwen3TTSConfig,
    *,
    max_batch_size: int,
) -> CodePredictorCudaExecutor:
    return CodePredictorCudaExecutor(
        model=model,
        layer0_embedding=layer0_embedding,
        config=config,
        max_batch_size=max_batch_size,
        driver=object(),
    )


def test_talker_prefill_and_decode_are_direct_bounded_jobs() -> None:
    torch.manual_seed(3)
    config = _config()
    model = Qwen3TTSTalkerModel(config).eval()
    job = _talker_executor(model, config)
    context = _AttentionContext()
    prefill = job.prefill(
        TalkerPrefillInput(
            attention_context=context,
            text_token_ids=torch.tensor([1, 2, 3]),
            codec_token_ids=torch.tensor([4, 5, 6]),
            codec_token_mask=torch.tensor([False, True, True]),
            last_token_indices=torch.tensor([2]),
            suppress_eos=torch.tensor([True]),
            sampling=_sampling(1, config.talker.vocab_size),
        )
    )
    assert prefill.tokens.shape == (1,)
    assert prefill.last_hidden.shape == (1, config.talker.hidden_size)
    assert context.advances == 1
    decode = job.decode(
        TalkerDecodeInput(
            attention_context=context,
            talker_step_embed=torch.zeros((1, config.talker.hidden_size)),
            text_token_ids=torch.tensor([config.tts_pad_token_id]),
            suppress_eos=torch.tensor([False]),
            sampling=_sampling(1, config.talker.vocab_size),
        )
    )
    assert decode.tokens.shape == (1,)
    assert context.advances == 2


def test_talker_eos_suppression_is_row_local_in_a_mixed_batch() -> None:
    config = _config()

    class _EosModel(torch.nn.Module):
        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        @staticmethod
        def initialize_projected_text_embedding_cache() -> None:
            return None

        @staticmethod
        def get_input_embeddings() -> torch.nn.Embedding:
            return torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)

        def forward(self, *, input_embeds: torch.Tensor, attention_context: object) -> torch.Tensor:
            del attention_context
            return input_embeds

        def codec_head(self, hidden: torch.Tensor) -> torch.Tensor:
            logits = hidden.new_full((hidden.shape[0], config.talker.vocab_size), -10.0)
            logits[:, config.talker.codec_eos_token_id - 1] = 9.0
            logits[:, config.talker.codec_eos_token_id] = 10.0
            return logits

    model = _EosModel()
    job = _talker_executor(model, config)
    context = _AttentionContext()
    result = job._forward_direct(
        attention_context=context,
        input_embeds=torch.zeros((2, config.talker.hidden_size)),
        last_token_indices=torch.tensor([0, 1]),
        suppress_eos=torch.tensor([True, False]),
        sampling=_sampling(2, config.talker.vocab_size),
    )

    eos = config.talker.codec_eos_token_id
    assert torch.isneginf(result.logits[0, eos])
    assert torch.isfinite(result.logits[1, eos])
    assert result.tokens.tolist() == [eos - 1, eos]


def test_code_predictor_runs_every_residual_group_in_one_job() -> None:
    torch.manual_seed(7)
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    model.consolidate_stacked_weights()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=2)
    values = CodePredictorInput(
        layer0_token=torch.tensor([1, 2]),
        past_hidden=torch.randn((2, config.talker.hidden_size)),
        temperature=torch.zeros(2),
        top_k=torch.zeros(2, dtype=torch.int32),
        top_p=torch.ones(2),
        seed=torch.tensor([3, 4]),
        offsets=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        num_code_groups=config.num_code_groups,
    )
    first = job.whole_frame(values)
    second = job.whole_frame(values)
    assert first.frames.shape == (2, config.num_code_groups)
    assert first.codec_sum.shape == (2, config.talker.hidden_size)
    torch.testing.assert_close(first.frames, second.frames)
    torch.testing.assert_close(first.codec_sum, second.codec_sum)


def test_code_predictor_rejects_a_shorter_caller_defined_frame() -> None:
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    model.consolidate_stacked_weights()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=1)

    with pytest.raises(ValueError, match="configured code groups"):
        job.whole_frame(
            CodePredictorInput(
                layer0_token=torch.tensor([1]),
                past_hidden=torch.randn((1, config.talker.hidden_size)),
                temperature=torch.zeros(1),
                top_k=torch.zeros(1, dtype=torch.int32),
                top_p=torch.ones(1),
                seed=torch.tensor([3]),
                offsets=torch.tensor([[1]]),
                num_code_groups=2,
            )
        )


def test_code_predictor_reuses_fully_overwritten_kv_without_clearing_it() -> None:
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=2)

    first = job._cache(2, torch.zeros(1))
    first.fill_(7)
    second = job._cache(2, torch.zeros(1))

    assert second.data_ptr() == first.data_ptr()
    assert torch.all(second == 7)


def test_code_predictor_scratch_cache_is_keyed_by_dtype_and_rejects_invalid_capacity() -> None:
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=2)

    fp32 = job._cache(1, torch.zeros(1, dtype=torch.float32))
    bf16 = job._cache(1, torch.zeros(1, dtype=torch.bfloat16))

    assert bf16.dtype == torch.bfloat16
    assert bf16.data_ptr() != fp32.data_ptr()
    with pytest.raises(ValueError, match="at least one row"):
        job._cache(0, torch.zeros(1))
    with pytest.raises(ValueError, match="exceeds"):
        job._cache(3, torch.zeros(1))

    with pytest.raises(ValueError, match="positive"):
        _code_predictor_executor(model, layer0, config, max_batch_size=0)


def test_talker_materialization_uses_projected_text_and_masked_codec_addition() -> None:
    config = _config()

    class _EmbeddingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
            self.codec = torch.nn.Embedding(20, 4)
            with torch.no_grad():
                self.codec.weight.copy_(torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4) * 10)

        def get_projected_text_embedding_cache(self) -> torch.Tensor:
            return self.text

        @property
        def device(self) -> torch.device:
            return self.codec.weight.device

        @staticmethod
        def initialize_projected_text_embedding_cache() -> None:
            return None

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return self.codec

    model = _EmbeddingModel()
    job = _talker_executor(model, config)
    text_ids = torch.tensor([1, 2, 3])
    codec_ids = torch.tensor([4, 5, 6])
    mask = torch.tensor([False, True, False])

    prefill = job.materialize_prefill(text_ids, codec_ids, mask)
    decode = job.materialize_decode(torch.ones((2, 4)), torch.tensor([7, 8]))

    expected_prefill = model.text.index_select(0, text_ids).clone()
    expected_prefill[1] += model.codec(codec_ids[1])
    torch.testing.assert_close(prefill, expected_prefill)
    torch.testing.assert_close(decode, torch.ones((2, 4)) + model.text.index_select(0, torch.tensor([7, 8])))


def test_code_predictor_routes_every_position_cache_slot_and_rng_offset() -> None:
    torch.manual_seed(11)
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    model.consolidate_stacked_weights()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=1)
    positions: list[torch.Tensor] = []
    cache_positions: list[int] = []
    sampled_offsets: list[torch.Tensor] = []
    original_forward = model.forward_depth_unrolled

    def traced_forward(hidden, position_ids, kv_cache, **kwargs):
        positions.append(position_ids.clone())
        cache_positions.append(kwargs["cache_pos"])
        return original_forward(hidden, position_ids, kv_cache, **kwargs)

    model.forward_depth_unrolled = traced_forward  # type: ignore[method-assign]

    def traced_sampler(logits, temperature, top_k, top_p, seed, offset):
        del temperature, top_k, top_p, seed
        sampled_offsets.append(offset.clone())
        return torch.remainder(offset, logits.shape[1]).to(torch.long)

    values = CodePredictorInput(
        layer0_token=torch.tensor([1]),
        past_hidden=torch.randn((1, config.talker.hidden_size)),
        temperature=torch.ones(1),
        top_k=torch.ones(1, dtype=torch.int32),
        top_p=torch.ones(1),
        seed=torch.tensor([23]),
        offsets=torch.tensor([[31, 47, 59]]),
        num_code_groups=config.num_code_groups,
        position_ids=torch.tensor([[5, 7, 11, 13]]),
    )

    result = job.whole_frame(values, capture_sampler=traced_sampler)

    assert cache_positions == [0, 1, 2, 3]
    assert [value.item() for value in positions] == [5, 7, 11, 13]
    assert [value.item() for value in sampled_offsets] == [31, 47, 59]
    assert result.frames.tolist() == [[1, 7, 11, 11]]


def test_code_predictor_rejects_negative_request_local_rng_addresses() -> None:
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    model.consolidate_stacked_weights()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=1)
    values = CodePredictorInput(
        layer0_token=torch.tensor([1]),
        past_hidden=torch.randn((1, config.talker.hidden_size)),
        temperature=torch.zeros(1),
        top_k=torch.ones(1, dtype=torch.int32),
        top_p=torch.ones(1),
        seed=torch.tensor([23]),
        offsets=torch.tensor([[31, -1, 59]]),
        num_code_groups=config.num_code_groups,
    )

    with pytest.raises(ValueError, match="offset"):
        job.whole_frame(values)


def test_code_predictor_does_not_mutate_typed_job_inputs() -> None:
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    model.consolidate_stacked_weights()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=1)
    values = CodePredictorInput(
        layer0_token=torch.tensor([1]),
        past_hidden=torch.randn((1, config.talker.hidden_size)),
        temperature=torch.zeros(1),
        top_k=torch.ones(1, dtype=torch.int32),
        top_p=torch.ones(1),
        seed=torch.tensor([23]),
        offsets=torch.tensor([[31, 47, 59]]),
        num_code_groups=config.num_code_groups,
        position_ids=torch.tensor([[5, 7, 11, 13]]),
    )
    snapshots = {
        name: value.clone()
        for name, value in {
            "layer0_token": values.layer0_token,
            "past_hidden": values.past_hidden,
            "temperature": values.temperature,
            "top_k": values.top_k,
            "top_p": values.top_p,
            "seed": values.seed,
            "offsets": values.offsets,
            "position_ids": values.position_ids,
        }.items()
        if value is not None
    }

    job.whole_frame(values)

    for name, expected in snapshots.items():
        actual = getattr(values, name)
        assert actual is not None
        torch.testing.assert_close(actual, expected)


def test_code_predictor_mixed_sampling_preserves_singleton_request_identity() -> None:
    torch.manual_seed(19)
    config = _config()
    model = Qwen3TTSCodePredictor(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.02, 0.02)
    model.consolidate_stacked_weights()
    layer0 = torch.nn.Embedding(config.talker.vocab_size, config.talker.hidden_size)
    job = _code_predictor_executor(model, layer0, config, max_batch_size=2)
    values = CodePredictorInput(
        layer0_token=torch.tensor([1, 2]),
        past_hidden=torch.randn((2, config.talker.hidden_size)),
        temperature=torch.tensor([0.0, 0.8]),
        top_k=torch.tensor([0, config.code_predictor.vocab_size], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.8]),
        seed=torch.tensor([101, 202]),
        offsets=torch.tensor([[11, 12, 13], [21, 22, 23]]),
        num_code_groups=config.num_code_groups,
    )

    batch = job.whole_frame(values)
    singletons = [
        job.whole_frame(
            CodePredictorInput(
                layer0_token=values.layer0_token[row : row + 1],
                past_hidden=values.past_hidden[row : row + 1],
                temperature=values.temperature[row : row + 1],
                top_k=values.top_k[row : row + 1],
                top_p=values.top_p[row : row + 1],
                seed=values.seed[row : row + 1],
                offsets=values.offsets[row : row + 1],
                num_code_groups=values.num_code_groups,
            )
        )
        for row in range(2)
    ]

    torch.testing.assert_close(batch.frames, torch.cat([result.frames for result in singletons]))
    torch.testing.assert_close(batch.codec_sum, torch.cat([result.codec_sum for result in singletons]))

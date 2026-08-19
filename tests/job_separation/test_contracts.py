from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from nari_qwen3_tts.executor.types import (
    CodecResult,
    CodePredictorInput,
    CodePredictorResult,
    TalkerDecodeInput,
    TalkerPrefillInput,
    TalkerResult,
    TalkerSamplingInput,
)
from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState


def _sampling(rows: int) -> TalkerSamplingInput:
    return TalkerSamplingInput(
        temperature=torch.full((rows,), 0.9),
        top_k=torch.full((rows,), 50, dtype=torch.int32),
        top_p=torch.ones(rows),
        repetition_penalty=torch.full((rows,), 1.05),
        seed=torch.arange(rows, dtype=torch.long),
        offsets=torch.arange(rows, dtype=torch.long),
        seen_token_mask=torch.zeros((rows, 32), dtype=torch.bool),
    )


def test_stage_inputs_are_typed_immutable_and_row_checked() -> None:
    value = TalkerDecodeInput(
        attention_context=object(),
        talker_step_embed=torch.zeros((2, 8)),
        text_token_ids=torch.zeros(2, dtype=torch.long),
        suppress_eos=torch.ones(2, dtype=torch.bool),
        sampling=_sampling(2),
    )
    with pytest.raises(FrozenInstanceError):
        value.attention_context = object()  # type: ignore[misc]

    with pytest.raises(ValueError, match="row count"):
        TalkerPrefillInput(
            attention_context=object(),
            text_token_ids=torch.zeros(3, dtype=torch.long),
            codec_token_ids=torch.zeros(3, dtype=torch.long),
            codec_token_mask=torch.zeros(3, dtype=torch.bool),
            last_token_indices=torch.tensor([2]),
            suppress_eos=torch.ones(2, dtype=torch.bool),
            sampling=_sampling(1),
        )


def test_code_predictor_requires_every_residual_rng_offset() -> None:
    with pytest.raises(ValueError, match="residual groups"):
        CodePredictorInput(
            layer0_token=torch.zeros(2, dtype=torch.long),
            past_hidden=torch.zeros((2, 8)),
            temperature=torch.ones(2),
            top_k=torch.ones(2, dtype=torch.int32),
            top_p=torch.ones(2),
            seed=torch.arange(2, dtype=torch.long),
            offsets=torch.zeros((2, 3), dtype=torch.long),
            num_code_groups=5,
        )


def test_stage_inputs_reject_empty_compute_batches() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        _sampling(0)

    with pytest.raises(ValueError, match="at least one row"):
        CodePredictorInput(
            layer0_token=torch.empty(0, dtype=torch.long),
            past_hidden=torch.empty((0, 8)),
            temperature=torch.empty(0),
            top_k=torch.empty(0, dtype=torch.int32),
            top_p=torch.empty(0),
            seed=torch.empty(0, dtype=torch.long),
            offsets=torch.empty((0, 3), dtype=torch.long),
            num_code_groups=4,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("top_k", torch.ones(2), "top_k"),
        ("seed", torch.ones(2, dtype=torch.int32), "seed"),
        ("offsets", torch.ones(2, dtype=torch.int32), "offsets"),
        ("seen_token_mask", torch.zeros((2, 32)), "seen_token_mask"),
    ],
)
def test_talker_sampling_requires_exact_tensor_kinds(field: str, replacement: torch.Tensor, message: str) -> None:
    values = {
        "temperature": torch.ones(2),
        "top_k": torch.ones(2, dtype=torch.int32),
        "top_p": torch.ones(2),
        "repetition_penalty": torch.ones(2),
        "seed": torch.ones(2, dtype=torch.long),
        "offsets": torch.ones(2, dtype=torch.long),
        "seen_token_mask": torch.zeros((2, 32), dtype=torch.bool),
    }
    values[field] = replacement

    with pytest.raises(TypeError, match=message):
        TalkerSamplingInput(**values)


def test_talker_stage_inputs_require_token_and_mask_dtypes() -> None:
    with pytest.raises(TypeError, match="text_token_ids"):
        TalkerDecodeInput(
            attention_context=object(),
            talker_step_embed=torch.zeros((1, 8)),
            text_token_ids=torch.zeros(1),
            suppress_eos=torch.ones(1, dtype=torch.bool),
            sampling=_sampling(1),
        )


@pytest.mark.parametrize("index", [-1, 3])
def test_talker_prefill_last_indices_must_address_the_packed_input(index: int) -> None:
    with pytest.raises(ValueError, match="last_token_indices"):
        TalkerPrefillInput(
            attention_context=object(),
            text_token_ids=torch.zeros(3, dtype=torch.long),
            codec_token_ids=torch.zeros(3, dtype=torch.long),
            codec_token_mask=torch.zeros(3, dtype=torch.bool),
            last_token_indices=torch.tensor([index]),
            suppress_eos=torch.zeros(1, dtype=torch.bool),
            sampling=_sampling(1),
        )
    with pytest.raises(TypeError, match="suppress_eos"):
        TalkerDecodeInput(
            attention_context=object(),
            talker_step_embed=torch.zeros((1, 8)),
            text_token_ids=torch.zeros(1, dtype=torch.long),
            suppress_eos=torch.ones(1),
            sampling=_sampling(1),
        )


def test_code_predictor_input_requires_fixed_tensor_kinds_and_group_count_type() -> None:
    base = {
        "layer0_token": torch.zeros(1, dtype=torch.long),
        "past_hidden": torch.zeros((1, 8)),
        "temperature": torch.ones(1),
        "top_k": torch.ones(1, dtype=torch.int32),
        "top_p": torch.ones(1),
        "seed": torch.zeros(1, dtype=torch.long),
        "offsets": torch.zeros((1, 3), dtype=torch.long),
        "num_code_groups": 4,
    }
    with pytest.raises(TypeError, match="layer0_token"):
        CodePredictorInput(**{**base, "layer0_token": torch.zeros(1)})
    with pytest.raises(TypeError, match="offsets"):
        CodePredictorInput(**{**base, "offsets": torch.zeros((1, 3), dtype=torch.int32)})
    with pytest.raises(TypeError, match="num_code_groups"):
        CodePredictorInput(**{**base, "num_code_groups": True})


@pytest.mark.parametrize(
    ("result", "field", "replacement"),
    [
        (
            TalkerResult(torch.zeros(1, dtype=torch.long), torch.zeros((1, 2)), torch.zeros((1, 3))),
            "tokens",
            torch.ones(1, dtype=torch.long),
        ),
        (
            CodePredictorResult(torch.zeros((1, 2), dtype=torch.long), torch.zeros((1, 2))),
            "frames",
            torch.ones((1, 2), dtype=torch.long),
        ),
        (CodecResult(torch.zeros((1, 2), dtype=torch.int16), None, False), "terminal", True),
    ],
)
def test_stage_results_are_immutable_completion_values(result: object, field: str, replacement: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, replacement)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (
            lambda: TalkerResult(torch.zeros(1), torch.zeros((1, 2)), torch.zeros((1, 3))),
            TypeError,
            "tokens",
        ),
        (
            lambda: TalkerResult(
                torch.zeros(2, dtype=torch.long),
                torch.zeros((1, 2)),
                torch.zeros((2, 3)),
            ),
            ValueError,
            "row",
        ),
        (
            lambda: CodePredictorResult(torch.zeros((1, 2)), torch.zeros((1, 2))),
            TypeError,
            "frames",
        ),
        (
            lambda: CodePredictorResult(
                torch.zeros((2, 2), dtype=torch.long),
                torch.zeros((1, 2)),
            ),
            ValueError,
            "row",
        ),
        (
            lambda: CodecResult(torch.zeros((1, 2)), None, False),
            TypeError,
            "PCM",
        ),
        (
            lambda: CodecResult(
                torch.zeros((2, 3), dtype=torch.int16),
                None,
                False,
                pcm_lengths=(3,),
            ),
            ValueError,
            "length",
        ),
        (
            lambda: CodecResult(
                torch.zeros((1, 3), dtype=torch.int16),
                None,
                False,
                pcm_lengths=(4,),
            ),
            ValueError,
            "length",
        ),
    ],
)
def test_stage_results_fail_closed_on_malformed_tensor_contracts(factory, error, message: str) -> None:
    with pytest.raises(error, match=message):
        factory()


def test_codec_result_requires_owned_tuple_state_and_length_values() -> None:
    pcm = torch.zeros((1, 2), dtype=torch.int16)
    with pytest.raises(TypeError, match="states"):
        CodecResult(pcm, [IncrementalCodecState()], False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="lengths"):
        CodecResult(pcm, None, False, pcm_lengths=[2])  # type: ignore[arg-type]

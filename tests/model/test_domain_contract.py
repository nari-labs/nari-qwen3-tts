from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from nari_qwen3_tts.contract.request import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_SPEAKERS,
    SynthesisRequest,
)
from nari_qwen3_tts.contract.rng import MAX_FUSED_RESIDUAL_TOP_K as MAX_RESIDUAL_TOP_K
from nari_qwen3_tts.model.capabilities import (
    FIXED_SILENT_BOOTSTRAP_FRAMES_BY_SPEAKER,
    QWEN3_TTS_CAPABILITIES,
)


def test_complete_custom_voice_domain_and_audio_facts_are_explicit() -> None:
    assert SUPPORTED_SPEAKERS == (
        "aiden",
        "dylan",
        "eric",
        "ono_anna",
        "ryan",
        "serena",
        "sohee",
        "uncle_fu",
        "vivian",
    )
    assert SUPPORTED_LANGUAGES == (
        "auto",
        "chinese",
        "english",
        "french",
        "german",
        "italian",
        "japanese",
        "korean",
        "portuguese",
        "russian",
        "spanish",
    )
    assert FIXED_SILENT_BOOTSTRAP_FRAMES_BY_SPEAKER == {
        speaker: 1 for speaker in SUPPORTED_SPEAKERS
    }
    assert QWEN3_TTS_CAPABILITIES.sample_rate == 24_000
    assert QWEN3_TTS_CAPABILITIES.samples_per_frame == 1_920
    assert QWEN3_TTS_CAPABILITIES.uses_request_fixed_sampling_seed is True


def test_max_output_tokens_preserve_minimum_without_rewriting_input() -> None:
    request = SynthesisRequest(text="one token request", max_new_tokens=1)
    assert request.max_new_tokens == 1
    assert request.effective_max_output_tokens == 2
    assert QWEN3_TTS_CAPABILITIES.max_output_tokens(request) == 2


def test_sampling_contract_is_request_local_and_complete() -> None:
    request = SynthesisRequest(
        text="sample",
        random_seed=73,
        do_sample=False,
        temperature=0.8,
        top_k=17,
        top_p=0.7,
        repetition_penalty=1.2,
        subtalker_dosample=False,
        subtalker_temperature=0.4,
        subtalker_top_k=9,
        subtalker_top_p=0.6,
    )
    assert request.temperature == 0.0
    assert request.talker_sampling == (False, 0.0, 17, 0.7, 1.2)
    assert request.subtalker_sampling == (False, 9, 0.6, 0.4)
    assert request.random_seed == 73


def test_request_is_immutable_and_preserves_full_vocabulary_top_k() -> None:
    request = SynthesisRequest(text="immutable", top_k=0)

    assert request.talker_sampling == (True, 0.9, 0, 1.0, 1.05)
    assert request.subtalker_sampling == (True, 50, 1.0, 0.9)
    with pytest.raises(FrozenInstanceError):
        request.random_seed = 9  # type: ignore[misc]


@pytest.mark.parametrize("subtalker_top_k", [0, MAX_RESIDUAL_TOP_K + 1])
def test_request_preserves_residual_top_k_domain_across_sampler_routes(
    subtalker_top_k: int,
) -> None:
    request = SynthesisRequest(text="Sampler domain", subtalker_top_k=subtalker_top_k)
    assert request.subtalker_top_k == subtalker_top_k


@pytest.mark.parametrize("subtalker_top_k", [0, MAX_RESIDUAL_TOP_K + 1])
def test_greedy_residual_decoding_accepts_any_top_k(subtalker_top_k: int) -> None:
    request = SynthesisRequest(
        text="greedy",
        subtalker_dosample=False,
        subtalker_temperature=0.0,
        subtalker_top_k=subtalker_top_k,
    )
    assert request.subtalker_top_k == subtalker_top_k


@pytest.mark.parametrize("speaker", SUPPORTED_SPEAKERS)
@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_default_custom_voice_domain_can_suppress_one_fixed_bootstrap_frame(
    speaker: str,
    language: str,
) -> None:
    request = SynthesisRequest(text="verified default", voice=speaker, language=language)
    assert QWEN3_TTS_CAPABILITIES.can_suppress_fixed_bootstrap_audio(request) is True


@pytest.mark.parametrize(
    "override",
    [
        {"skip_fixed_bootstrap_audio": False},
        {"instruct": "Speak softly."},
        {"temperature": 0.8},
        {"top_k": 49},
        {"top_p": 0.95},
        {"repetition_penalty": 1.0},
        {"subtalker_dosample": False},
        {"subtalker_temperature": 0.8},
        {"subtalker_top_k": 49},
        {"subtalker_top_p": 0.95},
        {"stream_chunk_schedule": (2, 4, 8, 12)},
        {"stream_chunk_frames": 12},
        {"stream_first_chunk_frames": 2},
        {"stream_steady_chunk_frames": 12},
    ],
)
def test_silent_bootstrap_suppression_fails_closed_outside_verified_contract(
    override: dict[str, object],
) -> None:
    request = SynthesisRequest(text="not the measured default", **override)
    assert QWEN3_TTS_CAPABILITIES.can_suppress_fixed_bootstrap_audio(request) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subtalker_dosample": 1},
        {"subtalker_top_k": True},
        {"subtalker_top_k": -1},
        {"subtalker_temperature": float("inf")},
        {"subtalker_top_p": 0.0},
        {"stream_chunk_schedule": ()},
        {"stream_chunk_schedule": (2, True)},
        {"stream_chunk_frames": 0},
    ],
)
def test_extended_model_request_contract_is_strict(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        SynthesisRequest(text="invalid", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": None},
        {"text": 1},
        {"text": " \t\n"},
        {"instruct": None},
        {"voice": None},
        {"voice": ""},
        {"voice": "unknown"},
        {"language": None},
        {"language": ""},
        {"language": "unknown"},
        {"non_streaming_mode": 1},
        {"random_seed": True},
        {"random_seed": -1},
        {"do_sample": 1},
        {"temperature": True},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"top_k": True},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"repetition_penalty": 0.0},
        {"max_new_tokens": True},
        {"max_new_tokens": 0},
        {"ignore_eos": 1},
        {"subtalker_dosample": 1},
        {"subtalker_temperature": float("inf")},
        {"subtalker_temperature": -0.1},
        {"subtalker_top_k": True},
        {"subtalker_top_k": -1},
        {"subtalker_top_p": 0.0},
        {"subtalker_top_p": 1.1},
        {"skip_fixed_bootstrap_audio": 1},
        {"stream_chunk_schedule": "2,4"},
        {"stream_chunk_schedule": (2, 0)},
        {"stream_chunk_schedule": (2, True)},
        {"stream_chunk_frames": True},
        {"stream_chunk_frames": 0},
        {"stream_first_chunk_frames": 0},
        {"stream_steady_chunk_frames": -1},
    ],
)
def test_every_request_field_family_fails_closed(kwargs: dict[str, object]) -> None:
    values = {"text": "strict request", **kwargs}
    with pytest.raises((TypeError, ValueError)):
        SynthesisRequest(**values)


@pytest.mark.parametrize("field", ["voice", "language"])
def test_all_supported_choices_normalize_case_space_and_hyphen(field: str) -> None:
    choices = SUPPORTED_SPEAKERS if field == "voice" else SUPPORTED_LANGUAGES
    for choice in choices:
        variant = f"  {choice.replace('_', '-').upper()}  "
        request = SynthesisRequest(text="normalization", **{field: variant})
        assert getattr(request, field) == choice


def test_seed_output_cap_and_input_mode_do_not_change_measured_bootstrap_prefix() -> None:
    for override in (
        {"random_seed": 99},
        {"max_new_tokens": 2},
        {"ignore_eos": True},
        {"non_streaming_mode": False},
    ):
        request = SynthesisRequest(text="prefix-invariant", **override)
        assert QWEN3_TTS_CAPABILITIES.can_suppress_fixed_bootstrap_audio(request) is True

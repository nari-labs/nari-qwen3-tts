"""Stable Qwen3-TTS model facts consumed by the synthesis engine."""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.contract.request import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_SPEAKERS,
    SynthesisRequest,
)

FIXED_SILENT_BOOTSTRAP_FRAMES_BY_SPEAKER = {
    speaker: 1 for speaker in SUPPORTED_SPEAKERS
}


@dataclass(frozen=True, slots=True)
class Qwen3TTSModelCapabilities:
    sample_rate: int = 24_000
    samples_per_frame: int = 1_920
    uses_request_fixed_sampling_seed: bool = True

    @staticmethod
    def max_output_tokens(request: SynthesisRequest) -> int:
        return request.effective_max_output_tokens

    @staticmethod
    def can_suppress_fixed_bootstrap_audio(request: SynthesisRequest) -> bool:
        return (
            request.skip_fixed_bootstrap_audio
            and not request.has_custom_stream_chunk_controls
            and FIXED_SILENT_BOOTSTRAP_FRAMES_BY_SPEAKER.get(request.voice) == 1
            and request.language in SUPPORTED_LANGUAGES
            and not request.instruct
            and request.talker_sampling == (True, 0.9, 50, 1.0, 1.05)
            and request.subtalker_sampling == (True, 50, 1.0, 0.9)
        )


QWEN3_TTS_CAPABILITIES = Qwen3TTSModelCapabilities()

__all__ = [
    "FIXED_SILENT_BOOTSTRAP_FRAMES_BY_SPEAKER",
    "QWEN3_TTS_CAPABILITIES",
    "Qwen3TTSModelCapabilities",
]

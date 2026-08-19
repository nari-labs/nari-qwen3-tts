"""Logical request-local RNG addressing shared by Talker and Code Predictor.

The address formula lives in :mod:`nari_qwen3_tts.contract.rng`; this module is the
typed execution-side view of the same space.
"""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.contract.rng import (
    PHILOX_CODEBOOK_STRIDE,
    QWEN3_TTS_CODEBOOKS,
    QWEN3_TTS_FRAME_STRIDE,
    logical_rng_offset,
)

TALKER_FRAME_OFFSET_STRIDE = QWEN3_TTS_FRAME_STRIDE


@dataclass(frozen=True, slots=True)
class TalkerCodebookAddress:
    frame_index: int
    codebook_index: int
    num_codebooks: int = QWEN3_TTS_CODEBOOKS
    philox_stride: int = PHILOX_CODEBOOK_STRIDE

    def __post_init__(self) -> None:
        if self.num_codebooks != QWEN3_TTS_CODEBOOKS or self.philox_stride != PHILOX_CODEBOOK_STRIDE:
            raise ValueError("Qwen3-TTS RNG layout is fixed at 16 codebooks and stride 32")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if (
            isinstance(self.codebook_index, bool)
            or not isinstance(self.codebook_index, int)
            or not 0 <= self.codebook_index < self.num_codebooks
        ):
            raise ValueError("codebook_index is outside the Qwen3-TTS codebook range")
        if self.num_codebooks < 1 or self.philox_stride < 1:
            raise ValueError("RNG address dimensions must be positive")

    @property
    def offset(self) -> int:
        return logical_rng_offset(self.frame_index, self.codebook_index)

    @property
    def is_talker(self) -> bool:
        return self.codebook_index == 0


__all__ = [
    "PHILOX_CODEBOOK_STRIDE",
    "QWEN3_TTS_CODEBOOKS",
    "TALKER_FRAME_OFFSET_STRIDE",
    "TalkerCodebookAddress",
]

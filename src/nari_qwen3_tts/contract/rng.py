"""Request-local logical sampling address space shared across stages."""

from __future__ import annotations

from enum import Enum

QWEN3_TTS_CODEBOOKS = 16
PHILOX_CODEBOOK_STRIDE = 32
QWEN3_TTS_FRAME_STRIDE = QWEN3_TTS_CODEBOOKS * PHILOX_CODEBOOK_STRIDE
MAX_FUSED_RESIDUAL_TOP_K = 64


class CodePredictorSamplerRoute(str, Enum):
    """Captured residual sampler selected before requests enter a batch."""

    FUSED = "fused"
    GENERAL = "general"


def code_predictor_sampler_route(
    *,
    temperature: float,
    top_k: int,
) -> CodePredictorSamplerRoute:
    """Select the fused/general route without making it batch-relative."""

    if temperature > 0 and not 1 <= top_k <= MAX_FUSED_RESIDUAL_TOP_K:
        return CodePredictorSamplerRoute.GENERAL
    return CodePredictorSamplerRoute.FUSED


def logical_rng_offset(frame_index: int, codebook_index: int) -> int:
    """Return the request-local FlashInfer offset for one logical draw."""

    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame index must be a non-negative integer")
    if (
        isinstance(codebook_index, bool)
        or not isinstance(codebook_index, int)
        or not 0 <= codebook_index < QWEN3_TTS_CODEBOOKS
    ):
        raise ValueError(f"codebook index must be in [0, {QWEN3_TTS_CODEBOOKS})")
    return frame_index * QWEN3_TTS_FRAME_STRIDE + codebook_index * PHILOX_CODEBOOK_STRIDE


__all__ = [
    "MAX_FUSED_RESIDUAL_TOP_K",
    "PHILOX_CODEBOOK_STRIDE",
    "QWEN3_TTS_CODEBOOKS",
    "QWEN3_TTS_FRAME_STRIDE",
    "CodePredictorSamplerRoute",
    "code_predictor_sampler_route",
    "logical_rng_offset",
]

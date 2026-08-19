"""Per-request rows and batch values crossing the Engine-to-Executor boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nari_qwen3_tts.contract.codec_state import IncrementalCodecState
from nari_qwen3_tts.contract.rng import CodePredictorSamplerRoute
from nari_qwen3_tts.executor._validate import require_scalar_int, require_scalar_number
from nari_qwen3_tts.executor.cache import PendingKVPublication
from nari_qwen3_tts.executor.types import TalkerResult


@dataclass(frozen=True, slots=True)
class TalkerSamplingExecutionRow:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int
    offset: int
    seen_token_mask: torch.Tensor | None

    def __post_init__(self) -> None:
        temperature = require_scalar_number("temperature", self.temperature)
        top_p = require_scalar_number("top_p", self.top_p)
        repetition_penalty = require_scalar_number(
            "repetition_penalty",
            self.repetition_penalty,
        )
        top_k = require_scalar_int("top_k", self.top_k)
        seed = require_scalar_int("seed", self.seed)
        offset = require_scalar_int("offset", self.offset)
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if self.seen_token_mask is not None:
            if self.seen_token_mask.ndim != 1:
                raise ValueError("seen_token_mask must be a vocabulary vector")
            if self.seen_token_mask.dtype is not torch.bool:
                raise TypeError("seen_token_mask must use torch.bool")


@dataclass(frozen=True, slots=True)
class TalkerPrefillExecutionRow:
    text_token_ids: torch.Tensor
    codec_token_ids: torch.Tensor
    codec_token_mask: torch.Tensor
    suppress_eos: bool
    sampling: TalkerSamplingExecutionRow

    def __post_init__(self) -> None:
        if not isinstance(self.suppress_eos, bool):
            raise TypeError("suppress_eos must be a boolean")
        if not isinstance(self.sampling, TalkerSamplingExecutionRow):
            raise TypeError("sampling must be a TalkerSamplingExecutionRow")
        if self.text_token_ids.ndim != 1 or self.text_token_ids.numel() < 1:
            raise ValueError("text_token_ids must be a non-empty token vector")
        if self.text_token_ids.dtype is not torch.long:
            raise TypeError("text_token_ids must use torch.long")
        if self.codec_token_ids.shape != self.text_token_ids.shape:
            raise ValueError("codec_token_ids must match text_token_ids")
        if self.codec_token_ids.dtype is not torch.long:
            raise TypeError("codec_token_ids must use torch.long")
        if self.codec_token_mask.shape != self.text_token_ids.shape:
            raise ValueError("codec_token_mask must match text_token_ids")
        if self.codec_token_mask.dtype is not torch.bool:
            raise TypeError("codec_token_mask must use torch.bool")
        tensors = (self.codec_token_ids, self.codec_token_mask, self.sampling.seen_token_mask)
        if any(value is not None and value.device != self.text_token_ids.device for value in tensors):
            raise ValueError("Talker prefill row tensors must share one device")


@dataclass(frozen=True, slots=True)
class TalkerDecodeExecutionRow:
    talker_step_embed: torch.Tensor
    text_token_id: torch.Tensor
    suppress_eos: bool
    sampling: TalkerSamplingExecutionRow

    def __post_init__(self) -> None:
        if not isinstance(self.suppress_eos, bool):
            raise TypeError("suppress_eos must be a boolean")
        if not isinstance(self.sampling, TalkerSamplingExecutionRow):
            raise TypeError("sampling must be a TalkerSamplingExecutionRow")
        if self.talker_step_embed.ndim != 1 or self.talker_step_embed.numel() < 1:
            raise ValueError("talker_step_embed must be a non-empty hidden vector")
        if not self.talker_step_embed.is_floating_point():
            raise TypeError("talker_step_embed must be floating point")
        if self.text_token_id.numel() != 1:
            raise ValueError("text_token_id must contain one token")
        if self.text_token_id.dtype is not torch.long:
            raise TypeError("text_token_id must use torch.long")
        if self.text_token_id.device.type != "cpu" and self.text_token_id.device != self.talker_step_embed.device:
            raise ValueError("Talker decode text must be on the host or execution device")
        mask = self.sampling.seen_token_mask
        if mask is not None and mask.device != self.talker_step_embed.device:
            raise ValueError("Talker decode seen-token mask must be on the execution device")


@dataclass(frozen=True, slots=True)
class TalkerPrefillRowsExecutionInput:
    request_ids: tuple[str, ...]
    rows: tuple[TalkerPrefillExecutionRow, ...]


@dataclass(frozen=True, slots=True)
class TalkerDecodeRowsExecutionInput:
    request_ids: tuple[str, ...]
    rows: tuple[TalkerDecodeExecutionRow, ...]
    reuse_attention_plan: bool = False


@dataclass(frozen=True, slots=True)
class TalkerExecutionResult:
    result: TalkerResult
    next_seen_token_masks: torch.Tensor
    next_sampling_offsets: torch.Tensor
    kv_publications: tuple[PendingKVPublication, ...]


@dataclass(frozen=True, slots=True)
class CodePredictorExecutionRow:
    layer0_token: torch.Tensor
    past_hidden: torch.Tensor
    temperature: float
    top_k: int
    top_p: float
    seed: int
    offsets: tuple[int, ...]
    position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:  # noqa: PLR0912 - scalar and tensor ingress is fail-closed
        temperature = require_scalar_number("temperature", self.temperature)
        top_p = require_scalar_number("top_p", self.top_p)
        top_k = require_scalar_int("top_k", self.top_k)
        seed = require_scalar_int("seed", self.seed)
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.offsets, tuple):
            raise TypeError("offsets must be a tuple")
        if not self.offsets:
            raise ValueError("offsets must cover residual codebooks")
        for offset in self.offsets:
            if require_scalar_int("offsets", offset) < 0:
                raise ValueError("offsets must be non-negative")
        if self.layer0_token.numel() != 1:
            raise ValueError("layer0_token must contain one token")
        if self.layer0_token.dtype is not torch.long:
            raise TypeError("layer0_token must use torch.long")
        if self.past_hidden.ndim != 1 or self.past_hidden.numel() < 1:
            raise ValueError("past_hidden must be a non-empty hidden vector")
        if not self.past_hidden.is_floating_point():
            raise TypeError("past_hidden must be floating point")
        if self.layer0_token.device != self.past_hidden.device:
            raise ValueError("Code Predictor row tensors must share one device")
        if self.position_ids is not None:
            if self.position_ids.ndim != 1 or self.position_ids.dtype is not torch.long:
                raise ValueError("position_ids must be a torch.long vector")
            if self.position_ids.device != self.layer0_token.device:
                raise ValueError("Code Predictor row tensors must share one device")


@dataclass(frozen=True, slots=True)
class CodePredictorRowsExecutionInput:
    rows: tuple[CodePredictorExecutionRow, ...]
    sampler_route: CodePredictorSamplerRoute = CodePredictorSamplerRoute.FUSED

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("Code Predictor rows must be a non-empty tuple")
        if not isinstance(self.sampler_route, CodePredictorSamplerRoute):
            raise TypeError("Code Predictor sampler route must be typed")


@dataclass(frozen=True, slots=True)
class CodecMetadataExecutionRow:
    """One request row for captured-work-free empty terminal metadata."""


@dataclass(frozen=True, slots=True)
class CodecExecutionRow:
    frames: tuple[torch.Tensor, ...]
    state: IncrementalCodecState | None
    visible_frames: int | None = None
    pcm_start_frame: int | None = None
    terminal: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple) or not self.frames:
            raise ValueError("Codec row frames must be a non-empty tuple")
        first = self.frames[0]
        for frame in self.frames:
            if frame.ndim != 1:
                raise ValueError("Codec row frame must be a codebook vector")
            if frame.dtype is not torch.long:
                raise TypeError("Codec row frame must use torch.long")
            if frame.shape != first.shape or frame.device != first.device:
                raise ValueError("Codec row frames must share shape and device")
        if self.state is not None and not isinstance(self.state, IncrementalCodecState):
            raise TypeError("Codec row state must be an IncrementalCodecState")
        for name in ("visible_frames", "pcm_start_frame"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"Codec row {name} must be a non-negative integer or None")
        if self.terminal is not None and not isinstance(self.terminal, bool):
            raise TypeError("Codec row terminal must be a boolean or None")


@dataclass(frozen=True, slots=True)
class CodecRowsExecutionInput:
    rows: tuple[CodecExecutionRow, ...]
    visible_frames: int
    pcm_start_frame: int = 0
    terminal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("Codec rows must be a non-empty tuple")
        for name in ("visible_frames", "pcm_start_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.terminal, bool):
            raise TypeError("terminal must be a boolean")


__all__ = [
    "CodecExecutionRow",
    "CodecMetadataExecutionRow",
    "CodecRowsExecutionInput",
    "CodePredictorExecutionRow",
    "CodePredictorRowsExecutionInput",
    "TalkerDecodeExecutionRow",
    "TalkerDecodeRowsExecutionInput",
    "TalkerExecutionResult",
    "TalkerPrefillExecutionRow",
    "TalkerPrefillRowsExecutionInput",
    "TalkerSamplingExecutionRow",
]

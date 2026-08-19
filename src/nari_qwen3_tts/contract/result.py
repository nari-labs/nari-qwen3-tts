"""Typed request-state proposals and stage completion contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from nari_qwen3_tts.contract.codec_state import IncrementalCodecState
from nari_qwen3_tts.contract.stage import SynthesisStage
from nari_qwen3_tts.contract.work import StageBatchRow


def _require_non_negative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class KVPublication:
    request_id: str
    pages: tuple[int, ...]
    length: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("KV publication requires a non-empty request ID")
        if not isinstance(self.pages, tuple):
            raise TypeError("KV publication pages must be an owned tuple")
        if any(isinstance(page, bool) or not isinstance(page, int) or page < 0 for page in self.pages):
            raise ValueError("KV publication page IDs must be non-negative integers")
        if len(set(self.pages)) != len(self.pages):
            raise ValueError("KV publication page IDs must be unique")
        _require_non_negative_integer("KV publication length", self.length)


@dataclass(frozen=True, slots=True)
class TalkerStateDelta:
    token: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    next_seen_token_mask: torch.Tensor
    next_sampling_offset: int
    kv: KVPublication | None

    def __post_init__(self) -> None:
        if self.token.ndim > 1 or self.token.numel() != 1 or self.token.dtype != torch.long:
            raise ValueError("Talker state token must contain one torch.long value")
        if self.hidden.ndim != 1 or not self.hidden.is_floating_point():
            raise ValueError("Talker state hidden value must be a floating-point vector")
        if self.logits.ndim != 1 or not self.logits.is_floating_point():
            raise ValueError("Talker state logits must be a floating-point vector")
        if self.next_seen_token_mask.shape != self.logits.shape or self.next_seen_token_mask.dtype != torch.bool:
            raise ValueError("Talker seen-token mask must be boolean and match logits")
        devices = {self.token.device, self.hidden.device, self.logits.device, self.next_seen_token_mask.device}
        if len(devices) != 1:
            raise ValueError("Talker state delta tensors must share one device")
        _require_non_negative_integer("Talker sampling offset", self.next_sampling_offset)
        if self.kv is not None and not isinstance(self.kv, KVPublication):
            raise TypeError("Talker KV proposal must be a KVPublication")


@dataclass(frozen=True, slots=True)
class CodePredictorStateDelta:
    frame: torch.Tensor
    codec_sum: torch.Tensor

    def __post_init__(self) -> None:
        if tuple(self.frame.shape) != (16,) or self.frame.dtype != torch.long:
            raise ValueError("Code Predictor state frame must contain 16 torch.long codebooks")
        if self.codec_sum.ndim != 1 or not self.codec_sum.is_floating_point():
            raise ValueError("Code Predictor codec sum must be a floating-point vector")
        if self.frame.device != self.codec_sum.device:
            raise ValueError("Code Predictor state delta tensors must share one device")


@dataclass(frozen=True, slots=True)
class CodecStateDelta:
    state: IncrementalCodecState | None
    consumed_frames: int
    visible_frames: int
    terminal: bool

    def __post_init__(self) -> None:
        if self.state is not None and not isinstance(self.state, IncrementalCodecState):
            raise TypeError("Codec successor must be an IncrementalCodecState")
        _require_non_negative_integer("Codec consumed frames", self.consumed_frames)
        _require_non_negative_integer("Codec visible frames", self.visible_frames)
        if self.visible_frames > self.consumed_frames:
            raise ValueError("Codec visible frames cannot exceed consumed frames")
        if not isinstance(self.terminal, bool):
            raise TypeError("Codec terminal must be a boolean")


RequestStateDelta: TypeAlias = TalkerStateDelta | CodePredictorStateDelta | CodecStateDelta


@dataclass(frozen=True, slots=True)
class StageBatchRowResult:
    row: StageBatchRow
    delta: RequestStateDelta | None

    def __post_init__(self) -> None:
        if not isinstance(self.row, StageBatchRow):
            raise TypeError("stage row result requires a StageBatchRow")
        if self.row.padding and self.delta is not None:
            raise ValueError("padding row cannot carry a state delta")
        if not self.row.padding and self.delta is None:
            raise ValueError("real row requires a state delta")


_DELTA_BY_STAGE = {
    SynthesisStage.TALKER_PREFILL: TalkerStateDelta,
    SynthesisStage.TALKER_DECODE: TalkerStateDelta,
    SynthesisStage.CODE_PREDICTOR: CodePredictorStateDelta,
    SynthesisStage.CODEC: CodecStateDelta,
}


@dataclass(frozen=True, slots=True)
class StageExecutionCompletion:
    batch_id: int
    stage: SynthesisStage
    rows: tuple[StageBatchRowResult, ...]
    error: BaseException | None = None

    def __post_init__(self) -> None:
        if isinstance(self.batch_id, bool) or not isinstance(self.batch_id, int) or self.batch_id < 1:
            raise ValueError("completion batch ID must be a positive integer")
        if not isinstance(self.stage, SynthesisStage):
            raise TypeError("completion stage must be a SynthesisStage")
        if self.error is not None:
            if not isinstance(self.error, BaseException):
                raise ValueError("failed completion requires a typed error")
            if self.rows:
                raise ValueError("failed completion cannot carry partial rows")
            return
        if not self.rows:
            raise ValueError("successful completion requires physical rows")
        expected = _DELTA_BY_STAGE[self.stage]
        if any(result.delta is not None and not isinstance(result.delta, expected) for result in self.rows):
            raise ValueError(f"{self.stage.value} completion requires {expected.__name__}")


__all__ = [
    "CodePredictorStateDelta",
    "CodecStateDelta",
    "KVPublication",
    "RequestStateDelta",
    "StageBatchRowResult",
    "StageExecutionCompletion",
    "TalkerStateDelta",
]

"""Canonical synthesis-stage and captured CUDA graph identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class RequestLane(str, Enum):
    GENERATION = "generation"
    CODEC = "codec"


class SynthesisStage(str, Enum):
    TALKER_PREFILL = "talker_prefill"
    TALKER_DECODE = "talker_decode"
    CODE_PREDICTOR = "code_predictor"
    CODEC = "codec"

    @property
    def lane(self) -> RequestLane:
        return RequestLane.CODEC if self is SynthesisStage.CODEC else RequestLane.GENERATION


class CodecExecutionMode(str, Enum):
    WHOLE_SEQUENCE = "whole_sequence"
    COLD = "cold"
    WARM = "warm"
    EMPTY = "empty"


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TalkerPrefillCaptureKey:
    capture_batch_size: int
    token_capacity: int
    capture_sequence_length: int | None

    def __post_init__(self) -> None:
        _require_positive_integer("capture batch size", self.capture_batch_size)
        _require_positive_integer("token capacity", self.token_capacity)
        if self.capture_sequence_length is not None:
            _require_positive_integer("capture sequence length", self.capture_sequence_length)


@dataclass(frozen=True, slots=True)
class TalkerDecodeCaptureKey:
    capture_batch_size: int

    def __post_init__(self) -> None:
        _require_positive_integer("capture batch size", self.capture_batch_size)


@dataclass(frozen=True, slots=True)
class CodePredictorCaptureKey:
    capture_batch_size: int

    def __post_init__(self) -> None:
        _require_positive_integer("capture batch size", self.capture_batch_size)


@dataclass(frozen=True, slots=True)
class CodecCaptureKey:
    mode: CodecExecutionMode
    model_frames: int
    capture_batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CodecExecutionMode):
            raise TypeError("Codec capture mode must be a CodecExecutionMode")
        if self.mode is CodecExecutionMode.EMPTY:
            raise ValueError("empty Codec work has no captured CUDA graph")
        _require_positive_integer("Codec model frames", self.model_frames)
        _require_positive_integer("capture batch size", self.capture_batch_size)


CudaGraphKey: TypeAlias = (
    TalkerPrefillCaptureKey
    | TalkerDecodeCaptureKey
    | CodePredictorCaptureKey
    | CodecCaptureKey
)


_CAPTURE_KEY_BY_STAGE = {
    SynthesisStage.TALKER_PREFILL: TalkerPrefillCaptureKey,
    SynthesisStage.TALKER_DECODE: TalkerDecodeCaptureKey,
    SynthesisStage.CODE_PREDICTOR: CodePredictorCaptureKey,
    SynthesisStage.CODEC: CodecCaptureKey,
}


@dataclass(frozen=True, slots=True)
class CudaGraphRef:
    """Stable lookup identity for one executor-owned captured graph."""

    stage: SynthesisStage
    key: CudaGraphKey

    def __post_init__(self) -> None:
        if not isinstance(self.stage, SynthesisStage):
            raise TypeError("CUDA graph stage must be a SynthesisStage")
        expected = _CAPTURE_KEY_BY_STAGE[self.stage]
        if not isinstance(self.key, expected):
            raise ValueError(f"{self.stage.value} requires {expected.__name__}")


__all__ = [
    "CodecCaptureKey",
    "CodecExecutionMode",
    "CodePredictorCaptureKey",
    "CudaGraphKey",
    "CudaGraphRef",
    "RequestLane",
    "SynthesisStage",
    "TalkerDecodeCaptureKey",
    "TalkerPrefillCaptureKey",
]

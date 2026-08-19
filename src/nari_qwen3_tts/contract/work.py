"""Token-free planning and batch execution contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

from nari_qwen3_tts.contract.rng import CodePredictorSamplerRoute
from nari_qwen3_tts.contract.stage import (
    CodecExecutionMode,
    CudaGraphRef,
    RequestLane,
    SynthesisStage,
)


def _require_non_negative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TalkerPrefillBatchCompatibility:
    sequence_length: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence_length, bool)
            or not isinstance(self.sequence_length, int)
            or self.sequence_length < 1
        ):
            raise ValueError("Talker prefill sequence length must be a positive integer")


@dataclass(frozen=True, slots=True)
class TalkerDecodeBatchCompatibility:
    pass


TALKER_DECODE_COMPATIBILITY = TalkerDecodeBatchCompatibility()


@dataclass(frozen=True, slots=True)
class CodePredictorBatchCompatibility:
    sampler_route: CodePredictorSamplerRoute = CodePredictorSamplerRoute.FUSED

    def __post_init__(self) -> None:
        if not isinstance(self.sampler_route, CodePredictorSamplerRoute):
            raise TypeError("Code Predictor sampler route must be typed")


@dataclass(frozen=True, slots=True)
class CodecBatchCompatibility:
    mode: CodecExecutionMode
    model_frames: int
    input_frames: int
    visible_frames: int
    pcm_start_frame: int
    producer_frames: int
    terminal: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CodecExecutionMode):
            raise TypeError("Codec mode must be a CodecExecutionMode")
        for name in (
            "model_frames",
            "input_frames",
            "visible_frames",
            "pcm_start_frame",
            "producer_frames",
        ):
            _require_non_negative_integer(name, getattr(self, name))
        if not isinstance(self.terminal, bool):
            raise TypeError("Codec terminal must be a boolean")
        if self.input_frames > self.model_frames:
            raise ValueError("Codec input frames cannot exceed model frames")
        if self.producer_frames > self.input_frames:
            raise ValueError("Codec producer frames cannot exceed input frames")
        if self.pcm_start_frame + self.visible_frames > self.input_frames:
            raise ValueError("Codec PCM window must fit the unpadded input")
        if self.model_frames == 0 and not self.terminal:
            raise ValueError("Only terminal metadata work may contain zero Codec frames")
        if self.mode is CodecExecutionMode.EMPTY and any(
            value != 0
            for value in (
                self.model_frames,
                self.input_frames,
                self.visible_frames,
                self.pcm_start_frame,
                self.producer_frames,
            )
        ):
            raise ValueError("empty Codec metadata work cannot carry frame data")
        if self.mode is not CodecExecutionMode.EMPTY and self.model_frames == 0:
            raise ValueError("non-empty Codec work requires model frames")


def codec_batch_compatible(
    anchor: CodecBatchCompatibility,
    candidate: CodecBatchCompatibility,
) -> bool:
    """Return whether two Codec rows share one exact captured execution shape."""

    if candidate == anchor:
        return True
    if (
        candidate.mode is not anchor.mode
        or candidate.model_frames != anchor.model_frames
        or candidate.terminal is not anchor.terminal
    ):
        return False

    def exact_model_input(value: CodecBatchCompatibility) -> bool:
        return value.input_frames == value.model_frames

    if anchor.mode in {CodecExecutionMode.WHOLE_SEQUENCE, CodecExecutionMode.COLD}:
        return exact_model_input(anchor) and exact_model_input(candidate)

    def padded_warm_terminal(value: CodecBatchCompatibility) -> bool:
        return (
            value.mode is CodecExecutionMode.WARM
            and value.terminal
            and value.model_frames > 0
            and value.pcm_start_frame == 0
            and value.input_frames == value.visible_frames == value.producer_frames
            and value.input_frames <= value.model_frames
        )

    return padded_warm_terminal(anchor) and padded_warm_terminal(candidate)


StageBatchCompatibility: TypeAlias = (
    TalkerPrefillBatchCompatibility
    | TalkerDecodeBatchCompatibility
    | CodePredictorBatchCompatibility
    | CodecBatchCompatibility
)

_COMPATIBILITY_BY_STAGE = {
    SynthesisStage.TALKER_PREFILL: TalkerPrefillBatchCompatibility,
    SynthesisStage.TALKER_DECODE: TalkerDecodeBatchCompatibility,
    SynthesisStage.CODE_PREDICTOR: CodePredictorBatchCompatibility,
    SynthesisStage.CODEC: CodecBatchCompatibility,
}


def _require_stage_compatibility(
    stage: SynthesisStage,
    compatibility: StageBatchCompatibility,
) -> None:
    expected = _COMPATIBILITY_BY_STAGE[stage]
    if not isinstance(compatibility, expected):
        raise ValueError(f"{stage.value} requires {expected.__name__}")


@dataclass(frozen=True, slots=True)
class ReadyStageWork:
    request_id: str
    stage: SynthesisStage
    version: int
    logical_step: int
    compatibility: StageBatchCompatibility
    admission_sequence: int
    ready_sequence: int = 0
    startup: bool = False
    deadline_s: float | None = None
    reserve_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("ready stage work requires a request ID")
        if not isinstance(self.stage, SynthesisStage):
            raise TypeError("ready stage work requires a SynthesisStage")
        for name in ("version", "logical_step", "admission_sequence", "ready_sequence"):
            _require_non_negative_integer(name, getattr(self, name))
        _require_stage_compatibility(self.stage, self.compatibility)
        if not isinstance(self.startup, bool):
            raise TypeError("startup must be a boolean")
        if self.deadline_s is not None and (
            isinstance(self.deadline_s, bool)
            or not isinstance(self.deadline_s, (int, float))
            or not math.isfinite(self.deadline_s)
        ):
            raise ValueError("playback deadline must be finite")
        if (
            isinstance(self.reserve_s, bool)
            or not isinstance(self.reserve_s, (int, float))
            or not math.isfinite(self.reserve_s)
            or self.reserve_s < 0
        ):
            raise ValueError("execution reserve must be finite and non-negative")
        if self.startup is (self.deadline_s is not None):
            raise ValueError("startup and playback deadline state are incoherent")

    @property
    def lane(self) -> RequestLane:
        return self.stage.lane

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.request_id,
            self.version,
            self.stage,
            self.logical_step,
            self.compatibility,
            self.admission_sequence,
        )


@dataclass(frozen=True, slots=True)
class StageBatchRow:
    physical_row: int
    request_id: str | None
    version: int | None
    logical_step: int
    compatibility: StageBatchCompatibility

    def __post_init__(self) -> None:
        _require_non_negative_integer("physical row", self.physical_row)
        _require_non_negative_integer("logical step", self.logical_step)
        if self.request_id is None:
            if self.version is not None:
                raise ValueError("padding row cannot carry a request version")
            return
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("a real row requires a non-empty request ID")
        _require_non_negative_integer("request version", self.version)

    @property
    def padding(self) -> bool:
        return self.request_id is None


@dataclass(frozen=True, slots=True)
class StageExecutionBatch:
    batch_id: int
    decision_id: int
    stage: SynthesisStage
    compatibility: StageBatchCompatibility
    capture: CudaGraphRef | None
    rows: tuple[StageBatchRow, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_id, bool)
            or not isinstance(self.batch_id, int)
            or self.batch_id < 1
            or isinstance(self.decision_id, bool)
            or not isinstance(self.decision_id, int)
            or self.decision_id < 1
        ):
            raise ValueError("batch and decision IDs must be positive integers")
        if not isinstance(self.stage, SynthesisStage):
            raise TypeError("execution batch requires a SynthesisStage")
        _require_stage_compatibility(self.stage, self.compatibility)
        if not self.rows:
            raise ValueError("execution batch requires physical rows")
        if tuple(row.physical_row for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("physical rows must be contiguous and ordered")
        if any(not row.padding for row in self.rows[self.logical_rows :]):
            raise ValueError("real rows cannot follow padding rows")
        for row in self.rows:
            _require_stage_compatibility(self.stage, row.compatibility)
        if self.capture is None:
            if not (
                self.stage is SynthesisStage.CODEC
                and isinstance(self.compatibility, CodecBatchCompatibility)
                and self.compatibility.mode is CodecExecutionMode.EMPTY
            ):
                raise ValueError("only empty Codec metadata work may omit a CUDA graph")
        elif self.capture.stage is not self.stage:
            raise ValueError("execution batch capture stage does not match")

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(row.request_id for row in self.rows if row.request_id is not None)

    @property
    def real_rows(self) -> tuple[StageBatchRow, ...]:
        return tuple(row for row in self.rows if not row.padding)

    @property
    def logical_rows(self) -> int:
        return sum(not row.padding for row in self.rows)

    @property
    def padding_rows(self) -> int:
        return len(self.rows) - self.logical_rows


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    decision_id: int
    batches: tuple[StageExecutionBatch, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_id, bool)
            or not isinstance(self.decision_id, int)
            or self.decision_id < 1
        ):
            raise ValueError("decision ID must be a positive integer")
        if not self.batches:
            raise ValueError("schedule decision requires execution batches")
        if any(batch.decision_id != self.decision_id for batch in self.batches):
            raise ValueError("schedule decision contains a batch from another decision")

    @property
    def selected_stage(self) -> SynthesisStage:
        return self.batches[0].stage

    @property
    def selected_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request_id
            for batch in self.batches
            for request_id in batch.request_ids
        )


__all__ = [
    "TALKER_DECODE_COMPATIBILITY",
    "CodePredictorBatchCompatibility",
    "CodecBatchCompatibility",
    "ReadyStageWork",
    "ScheduleDecision",
    "StageBatchCompatibility",
    "StageBatchRow",
    "StageExecutionBatch",
    "TalkerDecodeBatchCompatibility",
    "TalkerPrefillBatchCompatibility",
    "codec_batch_compatible",
]

"""Typed contracts for already-enqueued stage execution submissions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TypeAlias

from nari_qwen3_tts.contract.stage import SynthesisStage
from nari_qwen3_tts.contract.work import StageExecutionBatch
from nari_qwen3_tts.executor.cuda_graph import CudaSubmissionFence
from nari_qwen3_tts.executor.rows import (
    CodecExecutionRow,
    CodecMetadataExecutionRow,
    CodePredictorExecutionRow,
    TalkerDecodeExecutionRow,
    TalkerExecutionResult,
    TalkerPrefillExecutionRow,
)
from nari_qwen3_tts.executor.types import CodecResult, CodePredictorResult

StageExecutionInput: TypeAlias = (
    TalkerPrefillExecutionRow
    | TalkerDecodeExecutionRow
    | CodePredictorExecutionRow
    | CodecExecutionRow
    | CodecMetadataExecutionRow
)
StageExecutionResult: TypeAlias = TalkerExecutionResult | CodePredictorResult | CodecResult

_ROW_TYPES = (
    TalkerPrefillExecutionRow,
    TalkerDecodeExecutionRow,
    CodePredictorExecutionRow,
    CodecExecutionRow,
    CodecMetadataExecutionRow,
)


@dataclass(frozen=True, slots=True)
class StageExecutionInputs:
    """One batch's validated, row-ordered executor inputs."""

    batch_id: int
    rows: tuple[StageExecutionInput, ...]
    requires_host_finalize: bool = False
    reuse_attention_plan: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_id, bool)
            or not isinstance(self.batch_id, int)
            or self.batch_id < 1
        ):
            raise ValueError("stage execution batch ID must be a positive integer")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("stage execution inputs require non-empty rows")
        first_type = type(self.rows[0])
        if first_type not in _ROW_TYPES or any(type(row) is not first_type for row in self.rows):
            raise ValueError("stage execution rows must have one stage input type")
        if not isinstance(self.requires_host_finalize, bool):
            raise TypeError("requires_host_finalize must be a boolean")
        if self.requires_host_finalize and first_type is not TalkerDecodeExecutionRow:
            raise ValueError("host finalize is only valid for Talker decode")
        if not isinstance(self.reuse_attention_plan, bool):
            raise TypeError("reuse_attention_plan must be a boolean")
        if self.reuse_attention_plan and first_type is not TalkerDecodeExecutionRow:
            raise ValueError("attention-plan reuse is only valid for Talker decode")


@dataclass(frozen=True, slots=True)
class StageExecutionSubmission:
    """A stage result whose CUDA work has been enqueued but may be incomplete."""

    batch: StageExecutionBatch
    result: StageExecutionResult
    completion_fence: CudaSubmissionFence
    decision_fence: CudaSubmissionFence | None
    requires_host_finalize: bool

    def __post_init__(self) -> None:
        if not isinstance(self.batch, StageExecutionBatch):
            raise TypeError("submission batch must be a StageExecutionBatch")
        if not isinstance(self.completion_fence, CudaSubmissionFence):
            raise TypeError("submission completion fence must be a CudaSubmissionFence")
        if self.decision_fence is not None and not isinstance(
            self.decision_fence,
            CudaSubmissionFence,
        ):
            raise TypeError("submission decision fence must be a CudaSubmissionFence")
        if not isinstance(self.requires_host_finalize, bool):
            raise TypeError("requires_host_finalize must be a boolean")
        expected = {
            SynthesisStage.TALKER_PREFILL: TalkerExecutionResult,
            SynthesisStage.TALKER_DECODE: TalkerExecutionResult,
            SynthesisStage.CODE_PREDICTOR: CodePredictorResult,
            SynthesisStage.CODEC: CodecResult,
        }[self.batch.stage]
        if not isinstance(self.result, expected):
            raise TypeError(
                f"{self.batch.stage.value} submission requires {expected.__name__}"
            )
        if self.requires_host_finalize and self.batch.stage is not SynthesisStage.TALKER_DECODE:
            raise ValueError("host finalize is only valid for Talker decode")


class SubmissionFenceError(RuntimeError):
    """A host-visible submission fence failed while being polled or waited."""

    def __init__(self, submission: object, error: BaseException) -> None:
        super().__init__(str(error))
        self.submission = submission
        self.__cause__ = error


@dataclass(slots=True)
class _DecisionSubmissions:
    decision_id: int
    submissions: tuple[object, ...]

    @property
    def fence(self) -> object:
        return self.submissions[-1].decision_fence


class SubmissionWindow:
    """Bound and retire CUDA decisions without violating FIFO visibility."""

    def __init__(self, *, max_decisions: int = 2) -> None:
        if (
            isinstance(max_decisions, bool)
            or not isinstance(max_decisions, int)
            or max_decisions < 1
        ):
            raise ValueError("max_decisions must be a positive integer")
        self.max_decisions = max_decisions
        self._decisions: deque[_DecisionSubmissions] = deque()
        self._host_pending: list[object] = []

    @property
    def decision_ids(self) -> tuple[int, ...]:
        return tuple(item.decision_id for item in self._decisions)

    @property
    def can_submit(self) -> bool:
        return len(self._decisions) < self.max_decisions

    def record(self, *, decision_id: int, submissions: tuple[object, ...]) -> None:
        if isinstance(decision_id, bool) or not isinstance(decision_id, int) or decision_id < 1:
            raise ValueError("decision ID must be positive")
        if not isinstance(submissions, tuple) or not submissions:
            raise ValueError("decision submissions must be a non-empty tuple")
        if not self.can_submit:
            raise RuntimeError("submission window is full")
        if self._decisions and decision_id <= self._decisions[-1].decision_id:
            raise ValueError("decision IDs must be strictly increasing")
        for submission in submissions:
            completion_fence = getattr(submission, "completion_fence", None)
            if not all(
                callable(getattr(completion_fence, name, None))
                for name in ("ready", "wait")
            ):
                raise TypeError("submissions must carry a completion fence")
            if not isinstance(getattr(submission, "requires_host_finalize", None), bool):
                raise TypeError("submissions must declare host-finalize behavior")
        decision_fences = tuple(
            index
            for index, submission in enumerate(submissions)
            if getattr(submission, "decision_fence", None) is not None
        )
        if decision_fences != (len(submissions) - 1,):
            raise ValueError("only the final submission may carry the decision fence")
        decision_fence = submissions[-1].decision_fence
        if not all(
            callable(getattr(decision_fence, name, None))
            for name in ("ready", "wait")
        ):
            raise TypeError("final submission must carry a decision fence")
        self._decisions.append(
            _DecisionSubmissions(
                decision_id=decision_id,
                submissions=submissions,
            )
        )
        self._host_pending.extend(
            submission
            for submission in submissions
            if submission.requires_host_finalize
        )

    def poll_host_ready(self, *, block_oldest: bool) -> tuple[object, ...]:
        ready: list[object] = []
        wait_available = block_oldest
        remaining: list[object] = []
        for index, submission in enumerate(self._host_pending):
            try:
                if not submission.completion_fence.ready():
                    if not wait_available:
                        remaining.append(submission)
                        continue
                    submission.completion_fence.wait()
                    wait_available = False
            except BaseException as error:
                remaining.extend(self._host_pending[index + 1 :])
                self._host_pending = remaining
                raise SubmissionFenceError(submission, error) from error
            ready.append(submission)
        self._host_pending = remaining
        return tuple(ready)

    def reap_fences(self) -> tuple[int, ...]:
        retired: list[int] = []
        while self._decisions:
            decision = self._decisions[0]
            if not decision.fence.ready():
                break
            retired.append(self._decisions.popleft().decision_id)
        return tuple(retired)

    def wait_oldest(self) -> int | None:
        if not self._decisions:
            return None
        oldest = self._decisions[0]
        oldest.fence.wait()
        retired = self.reap_fences()
        return retired[0] if retired else None


__all__ = [
    "StageExecutionInput",
    "StageExecutionInputs",
    "StageExecutionResult",
    "StageExecutionSubmission",
    "SubmissionFenceError",
    "SubmissionWindow",
]

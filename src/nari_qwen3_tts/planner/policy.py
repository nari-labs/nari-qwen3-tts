"""Replaceable scheduling policies over immutable ready-stage work."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from nari_qwen3_tts.contract.stage import SynthesisStage
from nari_qwen3_tts.contract.work import (
    CodecBatchCompatibility,
    ReadyStageWork,
    codec_batch_compatible,
)

RR_STAGE_ORDER = (
    SynthesisStage.TALKER_PREFILL,
    SynthesisStage.TALKER_DECODE,
    SynthesisStage.CODEC,
    SynthesisStage.CODE_PREDICTOR,
)
STAGE_RANK = {stage: rank for rank, stage in enumerate(RR_STAGE_ORDER)}


class SelectionReason(str, Enum):
    ROUND_ROBIN = "round_robin"


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    selected_stage: SynthesisStage
    rr_counterfactual: SynthesisStage
    reason: str
    policy_inputs: tuple[tuple[str, object], ...] = ()
    anchor_request_id: str | None = None
    priority_request_ids: tuple[str, ...] = ()
    cohort_request_ids: tuple[str, ...] = ()

    @property
    def overrode_round_robin(self) -> bool:
        return self.selected_stage is not self.rr_counterfactual


class SchedulingPolicy(Protocol):
    name: str

    def choose(
        self,
        eligible: tuple[ReadyStageWork, ...],
        *,
        decision_sequence: int,
        now_s: float,
    ) -> PolicyEvaluation: ...

    def commit(self, evaluation: PolicyEvaluation, *, decision_sequence: int) -> None: ...


class RoundRobinPolicy:
    """Least-recent stage service with a fixed native tie order."""

    name = SelectionReason.ROUND_ROBIN.value

    def __init__(self) -> None:
        self._last_service: dict[SynthesisStage, int] = {}
        self._first_pass_precedence_needs_yield = False

    def select(self, eligible: tuple[ReadyStageWork, ...]) -> SynthesisStage:
        if not eligible:
            raise ValueError("cannot select from empty eligible work")
        stages = {work.stage for work in eligible}
        return min(
            stages,
            key=lambda stage: (
                self._last_service.get(stage, 0),
                STAGE_RANK[stage],
            ),
        )

    def choose(
        self,
        eligible: tuple[ReadyStageWork, ...],
        *,
        decision_sequence: int,
        now_s: float,
    ) -> PolicyEvaluation:
        del decision_sequence, now_s
        rr_stage = self.select(eligible)
        if self._first_pass_precedence_needs_yield:
            return PolicyEvaluation(rr_stage, rr_stage, self.name)

        first_codec_ids = tuple(
            work.request_id
            for work in eligible
            if work.stage is SynthesisStage.CODEC and work.startup and work.logical_step == 0
        )
        if not first_codec_ids:
            return PolicyEvaluation(rr_stage, rr_stage, self.name)
        first_codec_set = set(first_codec_ids)
        precedence_eligible = tuple(
            work
            for work in eligible
            if not (
                work.stage is SynthesisStage.TALKER_DECODE
                and work.request_id in first_codec_set
            )
            and not (
                work.stage is SynthesisStage.CODEC
                and work.request_id not in first_codec_set
            )
        )
        selected = self.select(precedence_eligible)
        if selected is not SynthesisStage.CODEC:
            return PolicyEvaluation(selected, rr_stage, "first_pass_precedence")
        return PolicyEvaluation(
            selected,
            rr_stage,
            "first_pass_precedence",
            cohort_request_ids=first_codec_ids,
        )

    def commit(self, evaluation: PolicyEvaluation, *, decision_sequence: int) -> None:
        self._last_service[evaluation.selected_stage] = decision_sequence
        if self._first_pass_precedence_needs_yield:
            self._first_pass_precedence_needs_yield = False
        elif (
            evaluation.reason == "first_pass_precedence"
            and evaluation.selected_stage is SynthesisStage.CODEC
        ):
            self._first_pass_precedence_needs_yield = True

    def checkpoint(self) -> tuple[tuple[SynthesisStage, int], ...]:
        return tuple(sorted(self._last_service.items(), key=lambda item: STAGE_RANK[item[0]]))


_STARTUP_STAGE_RANK = {
    SynthesisStage.CODEC: 0,
    SynthesisStage.TALKER_PREFILL: 1,
    SynthesisStage.CODE_PREDICTOR: 2,
    SynthesisStage.TALKER_DECODE: 3,
}


class DeadlineAwarePolicy:
    """Urgency/startup ranking over unchanged ready work with an RR shadow."""

    name = "deadline_aware"

    def __init__(self, *, lead_s: float, round_robin: RoundRobinPolicy | None = None) -> None:
        if isinstance(lead_s, bool) or not isinstance(lead_s, (int, float)):
            raise TypeError("deadline-aware lead_s must be a number")
        if not math.isfinite(lead_s) or lead_s < 0:
            raise ValueError("deadline-aware lead_s must be finite and non-negative")
        self.lead_s = float(lead_s)
        self.round_robin = round_robin or RoundRobinPolicy()

    @staticmethod
    def _urgent_codec_key(work: ReadyStageWork) -> tuple[object, ...]:
        deadline = float("inf") if work.deadline_s is None else work.deadline_s
        return deadline, deadline - work.reserve_s, work.request_id

    @staticmethod
    def _pressing_key(work: ReadyStageWork) -> tuple[object, ...]:
        deadline = float("inf") if work.deadline_s is None else work.deadline_s
        return deadline, STAGE_RANK[work.stage], work.request_id

    @staticmethod
    def _completion_deadline_key(work: ReadyStageWork) -> tuple[object, ...]:
        deadline = float("inf") if work.deadline_s is None else work.deadline_s
        return deadline, work.request_id

    def choose(
        self,
        eligible: tuple[ReadyStageWork, ...],
        *,
        decision_sequence: int,
        now_s: float,
    ) -> PolicyEvaluation:
        del decision_sequence
        rr_stage = self.round_robin.select(eligible)
        urgent = tuple(
            work
            for work in eligible
            if (
                work.stage is SynthesisStage.CODEC
                and not work.startup
                and work.deadline_s is not None
                and now_s >= work.deadline_s - work.reserve_s
            )
        )
        if urgent:
            anchor = min(urgent, key=self._urgent_codec_key)
            priority = tuple(
                work.request_id
                for work in sorted(eligible, key=self._completion_deadline_key)
                if work.stage is anchor.stage
                and isinstance(work.compatibility, CodecBatchCompatibility)
                and isinstance(anchor.compatibility, CodecBatchCompatibility)
                and codec_batch_compatible(anchor.compatibility, work.compatibility)
            )
            return PolicyEvaluation(
                anchor.stage,
                rr_stage,
                "deadline_aware_urgent_codec",
                (
                    ("now_s", now_s),
                    ("deadline_s", anchor.deadline_s),
                    ("reserve_s", anchor.reserve_s),
                ),
                anchor.request_id,
                priority,
            )

        startup = tuple(work for work in eligible if work.startup)
        if startup:
            anchor = min(
                startup,
                key=lambda work: (
                    _STARTUP_STAGE_RANK[work.stage],
                    work.admission_sequence,
                    work.ready_sequence,
                    work.request_id,
                ),
            )
            return PolicyEvaluation(
                anchor.stage,
                rr_stage,
                "deadline_aware_startup",
                (("startup_stage_rank", _STARTUP_STAGE_RANK[anchor.stage]),),
                anchor.request_id,
            )

        pressing = tuple(
            work
            for work in eligible
            if work.deadline_s is not None and now_s >= work.deadline_s - self.lead_s
        )
        if pressing:
            anchor = min(pressing, key=self._pressing_key)
            priority = tuple(
                work.request_id
                for work in sorted(pressing, key=self._pressing_key)
                if work.stage is anchor.stage
                and (
                    isinstance(work.compatibility, CodecBatchCompatibility)
                    and isinstance(anchor.compatibility, CodecBatchCompatibility)
                    and codec_batch_compatible(anchor.compatibility, work.compatibility)
                    if anchor.stage is SynthesisStage.CODEC
                    else True
                )
            )
            return PolicyEvaluation(
                anchor.stage,
                rr_stage,
                self.name,
                (
                    ("now_s", now_s),
                    ("deadline_s", anchor.deadline_s),
                    ("lead_s", self.lead_s),
                ),
                anchor.request_id,
                priority,
            )
        return PolicyEvaluation(rr_stage, rr_stage, RoundRobinPolicy.name)

    def commit(self, evaluation: PolicyEvaluation, *, decision_sequence: int) -> None:
        self.round_robin.commit(evaluation, decision_sequence=decision_sequence)


__all__ = [
    "PolicyEvaluation",
    "RR_STAGE_ORDER",
    "RoundRobinPolicy",
    "STAGE_RANK",
    "SchedulingPolicy",
    "SelectionReason",
    "DeadlineAwarePolicy",
]

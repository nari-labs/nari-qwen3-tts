"""Read-only request readiness for fixed Qwen3-TTS synthesis stages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from nari_qwen3_tts.contract.request import AdmittedRequest
from nari_qwen3_tts.contract.rng import code_predictor_sampler_route
from nari_qwen3_tts.contract.stage import CudaGraphRef, SynthesisStage
from nari_qwen3_tts.contract.work import (
    TALKER_DECODE_COMPATIBILITY,
    CodecBatchCompatibility,
    CodePredictorBatchCompatibility,
    ReadyStageWork,
    ScheduleDecision,
    StageBatchCompatibility,
    StageBatchRow,
    StageExecutionBatch,
    TalkerPrefillBatchCompatibility,
    codec_batch_compatible,
)
from nari_qwen3_tts.planner.catalog import CaptureCatalog, PhysicalBatchSlice
from nari_qwen3_tts.planner.policy import (
    STAGE_RANK,
    PolicyEvaluation,
    SchedulingPolicy,
)


def _phase_name(value: object) -> str:
    phase = getattr(value, "value", value)
    if not isinstance(phase, str):
        raise TypeError("request phase must be a string enum")
    return phase


class _GenerationStateView(Protocol):
    phase: object
    generation_step: int
    claim_token: int | None


class _CodecStateView(Protocol):
    chunk_index: int
    playback_started_at_s: float | None
    emitted_duration_s: float
    execution_reserve_s: float
    claim_token: int | None
    ready_compatibility: CodecBatchCompatibility | None


class PlanningRequestView(Protocol):
    """Read-only request surface consumed by readiness derivation."""

    request_id: str
    input: AdmittedRequest | None
    generation: _GenerationStateView
    codec: _CodecStateView
    cancel_requested: bool
    admission_sequence: int

    def version_for(self, stage: SynthesisStage) -> int: ...


def _request_input(state: PlanningRequestView) -> AdmittedRequest:
    value = state.input
    if not isinstance(value, AdmittedRequest):
        raise RuntimeError("request lacks its immutable admitted input")
    return value


class Planner:
    """Build immutable ready work without claiming or mutating request state."""

    def __init__(
        self,
        *,
        catalog: CaptureCatalog | None = None,
        policy: SchedulingPolicy | None = None,
    ) -> None:
        self.catalog = catalog
        self.policy = policy
        self._snapshot_sequence = 0
        self._active: dict[tuple[object, ...], int] = {}
        self._batch_sequence = 0
        self._pending: dict[int, tuple[ScheduleDecision, PolicyEvaluation]] = {}
        self._observations: dict[int, PlanningObservation] = {}

    @staticmethod
    def _candidate(
        state: PlanningRequestView,
        stage: SynthesisStage,
        compatibility: StageBatchCompatibility,
    ) -> ReadyStageWork:
        codec = state.codec
        generation = state.generation
        logical_step = codec.chunk_index if stage is SynthesisStage.CODEC else generation.generation_step
        playback_started_at_s = codec.playback_started_at_s
        return ReadyStageWork(
            request_id=state.request_id,
            stage=stage,
            version=state.version_for(stage),
            logical_step=logical_step,
            compatibility=compatibility,
            admission_sequence=state.admission_sequence,
            startup=playback_started_at_s is None,
            deadline_s=(
                None
                if playback_started_at_s is None
                else playback_started_at_s + codec.emitted_duration_s
            ),
            reserve_s=codec.execution_reserve_s if stage is SynthesisStage.CODEC else 0.0,
        )

    def candidates(
        self,
        states: Sequence[PlanningRequestView],
        *,
        now_s: float,
    ) -> tuple[ReadyStageWork, ...]:
        if isinstance(now_s, bool) or not isinstance(now_s, (int, float)) or not math.isfinite(now_s):
            raise ValueError("readiness time must be a finite number")
        candidates: list[ReadyStageWork] = []
        for state in states:
            if state.cancel_requested:
                continue
            generation = state.generation
            if generation.claim_token is None:
                generation_phase = _phase_name(generation.phase)
                if generation_phase == "talker_prefill":
                    prompt = _request_input(state).talker_input
                    candidates.append(
                        self._candidate(
                            state,
                            SynthesisStage.TALKER_PREFILL,
                            TalkerPrefillBatchCompatibility(prompt.sequence_length),
                        )
                    )
                elif generation_phase == "code_predictor":
                    request = _request_input(state).request
                    candidates.append(
                        self._candidate(
                            state,
                            SynthesisStage.CODE_PREDICTOR,
                            CodePredictorBatchCompatibility(
                                sampler_route=code_predictor_sampler_route(
                                    temperature=(
                                        request.subtalker_temperature
                                        if request.subtalker_dosample
                                        else 0.0
                                    ),
                                    top_k=request.subtalker_top_k,
                                )
                            ),
                        )
                    )
                elif generation_phase == "talker_decode":
                    continuation = _request_input(state).talker_input.continuation
                    if continuation.has_token(generation.generation_step):
                        candidates.append(
                            self._candidate(
                                state,
                                SynthesisStage.TALKER_DECODE,
                                TALKER_DECODE_COMPATIBILITY,
                            )
                        )
            codec = state.codec
            if codec.claim_token is None and codec.ready_compatibility is not None:
                candidates.append(
                    self._candidate(
                        state,
                        SynthesisStage.CODEC,
                        codec.ready_compatibility,
                    )
                )
        return tuple(candidates)

    def _capture(
        self,
        candidates: tuple[ReadyStageWork, ...],
    ) -> tuple[int, tuple[ReadyStageWork, ...]]:
        self._snapshot_sequence += 1
        current: dict[tuple[object, ...], int] = {}
        ready: list[ReadyStageWork] = []
        for work in sorted(
            candidates,
            key=lambda item: (
                item.admission_sequence,
                item.request_id,
                STAGE_RANK[item.stage],
                item.logical_step,
            ),
        ):
            ready_sequence = self._active.get(work.identity, self._snapshot_sequence)
            current[work.identity] = ready_sequence
            ready.append(replace(work, ready_sequence=ready_sequence))
        self._active = current
        return self._snapshot_sequence, tuple(ready)

    @staticmethod
    def _ordered(items: tuple[ReadyStageWork, ...]) -> tuple[ReadyStageWork, ...]:
        return tuple(
            sorted(
                items,
                key=lambda work: (
                    work.ready_sequence,
                    work.admission_sequence,
                    work.request_id,
                ),
            )
        )

    def _cohort(
        self,
        eligible: tuple[ReadyStageWork, ...],
        evaluation: PolicyEvaluation,
    ) -> tuple[ReadyStageWork, ...]:
        stage_items = self._ordered(
            tuple(work for work in eligible if work.stage is evaluation.selected_stage)
        )
        if not stage_items:
            raise ValueError("policy selected stage is not eligible")
        eligible_ids = {work.request_id for work in stage_items}
        if evaluation.anchor_request_id is not None and evaluation.anchor_request_id not in eligible_ids:
            raise ValueError("policy anchor is not eligible for the selected stage")
        for name, request_ids in (
            ("priority", evaluation.priority_request_ids),
            ("cohort", evaluation.cohort_request_ids),
        ):
            if len(set(request_ids)) != len(request_ids):
                raise ValueError(f"policy {name} request IDs must be unique")
            if any(request_id not in eligible_ids for request_id in request_ids):
                raise ValueError(f"policy {name} request is not eligible for the selected stage")
        if evaluation.cohort_request_ids:
            requested = set(evaluation.cohort_request_ids)
            stage_items = tuple(work for work in stage_items if work.request_id in requested)
        anchor = next(
            (
                work
                for work in stage_items
                if work.request_id == evaluation.anchor_request_id
            ),
            stage_items[0],
        )
        if evaluation.selected_stage is SynthesisStage.CODEC:
            if not isinstance(anchor.compatibility, CodecBatchCompatibility):
                raise TypeError("Codec policy anchor requires Codec compatibility")
            cohort = tuple(
                work
                for work in stage_items
                if isinstance(work.compatibility, CodecBatchCompatibility)
                and codec_batch_compatible(anchor.compatibility, work.compatibility)
            )
        elif evaluation.selected_stage is SynthesisStage.CODE_PREDICTOR:
            cohort = tuple(work for work in stage_items if work.compatibility == anchor.compatibility)
        else:
            cohort = stage_items
        if evaluation.anchor_request_id is None:
            return cohort
        priority = {
            request_id: rank
            for rank, request_id in enumerate(evaluation.priority_request_ids)
        }
        return tuple(
            sorted(
                cohort,
                key=lambda work: (
                    0 if work.request_id in priority else 1,
                    priority.get(work.request_id, 0),
                    0 if work is anchor else 1,
                    work.ready_sequence,
                    work.admission_sequence,
                    work.request_id,
                ),
            )
        )

    def _lower(
        self,
        selected: tuple[ReadyStageWork, ...],
    ) -> tuple[PhysicalBatchSlice, ...]:
        if self.catalog is None:
            raise RuntimeError("Planner capture catalog is not configured")
        stage = selected[0].stage
        if stage is SynthesisStage.TALKER_PREFILL:
            return self.catalog.lower_talker_prefill(
                tuple(
                    work.compatibility.sequence_length
                    for work in selected
                    if isinstance(work.compatibility, TalkerPrefillBatchCompatibility)
                )
            )
        if stage is SynthesisStage.TALKER_DECODE:
            return self.catalog.lower_talker_decode(len(selected))
        if stage is SynthesisStage.CODE_PREDICTOR:
            return self.catalog.lower_code_predictor(len(selected))
        compatibility = selected[0].compatibility
        if not isinstance(compatibility, CodecBatchCompatibility):
            raise TypeError("Codec selection requires Codec compatibility")
        if compatibility.model_frames == 0:
            return (self.catalog.lower_empty_terminal(len(selected)),)
        return self.catalog.lower_codec(
            compatibility.mode,
            model_frames=compatibility.model_frames,
            logical_rows=len(selected),
        )

    def _batch(
        self,
        *,
        decision_id: int,
        selected: tuple[ReadyStageWork, ...],
        physical: PhysicalBatchSlice,
    ) -> StageExecutionBatch:
        logical = selected[physical.logical_start : physical.logical_stop]
        self._batch_sequence += 1
        rows = [
            StageBatchRow(
                physical_row=index,
                request_id=work.request_id,
                version=work.version,
                logical_step=work.logical_step,
                compatibility=work.compatibility,
            )
            for index, work in enumerate(logical)
        ]
        rows.extend(
            StageBatchRow(
                physical_row=index,
                request_id=None,
                version=None,
                logical_step=logical[0].logical_step,
                compatibility=logical[-1].compatibility,
            )
            for index in range(len(logical), physical.capture_batch_size)
        )
        capture = (
            None
            if physical.key is None
            else CudaGraphRef(logical[0].stage, physical.key)
        )
        return StageExecutionBatch(
            batch_id=self._batch_sequence,
            decision_id=decision_id,
            stage=logical[0].stage,
            compatibility=logical[0].compatibility,
            capture=capture,
            rows=tuple(rows),
        )

    def plan(
        self,
        candidates: tuple[ReadyStageWork, ...],
        *,
        now_s: float,
        decision_id: int,
        available_rows: int,
        record_observation: bool = True,
    ) -> ScheduleDecision:
        if not candidates:
            raise ValueError("cannot schedule empty ready work")
        if self.policy is None or self.catalog is None:
            raise RuntimeError("Planner policy and capture catalog must be configured")
        if (
            isinstance(now_s, bool)
            or not isinstance(now_s, (int, float))
            or not math.isfinite(now_s)
        ):
            raise ValueError("planning time must be a finite number")
        if isinstance(decision_id, bool) or not isinstance(decision_id, int) or decision_id < 1:
            raise ValueError("decision ID must be positive")
        if decision_id in self._pending or decision_id in self._observations:
            raise ValueError("decision ID was already planned")
        if isinstance(available_rows, bool) or not isinstance(available_rows, int) or available_rows < 1:
            raise ValueError("available rows must be positive")
        if not isinstance(record_observation, bool):
            raise TypeError("record_observation must be a boolean")
        snapshot_sequence, ready = self._capture(candidates)
        evaluation = self.policy.choose(
            ready,
            decision_sequence=decision_id,
            now_s=now_s,
        )
        cohort = self._cohort(ready, evaluation)
        selection_limit = available_rows
        if (
            evaluation.selected_stage is SynthesisStage.CODEC
            and evaluation.reason
            in {
                "deadline_aware_urgent_codec",
                "deadline_aware_startup",
                "deadline_aware",
            }
        ):
            compatibility = cohort[0].compatibility
            if not isinstance(compatibility, CodecBatchCompatibility):
                raise TypeError("Codec cohort requires Codec compatibility")
            selection_limit = min(
                selection_limit,
                self.catalog.codec_batch_capacity(
                    compatibility.mode,
                    model_frames=compatibility.model_frames,
                ),
            )
        selected = cohort[:selection_limit]
        batches = tuple(
            self._batch(decision_id=decision_id, selected=selected, physical=physical)
            for physical in self._lower(selected)
        )
        decision = ScheduleDecision(decision_id, batches)
        self._pending[decision_id] = (decision, evaluation)
        if record_observation:
            selected_identities = {work.identity for work in selected}
            cohort_identities = {work.identity for work in cohort}
            waits = tuple(
                PlanningWaitReason(
                    request_id=work.request_id,
                    stage=work.stage,
                    reason=(
                        PlanningWaitReasonCode.RESOURCE_EXHAUSTED
                        if work.identity in cohort_identities
                        else (
                            PlanningWaitReasonCode.INCOMPATIBLE_COHORT
                            if work.stage is evaluation.selected_stage
                            else PlanningWaitReasonCode.RR_STAGE_NOT_SELECTED
                        )
                    ),
                    wait_decisions=max(0, snapshot_sequence - work.ready_sequence),
                )
                for work in ready
                if work.identity not in selected_identities
            )
            self._observations[decision_id] = PlanningObservation(
                snapshot_sequence=snapshot_sequence,
                ready=ready,
                eligible=ready,
                selected=selected,
                evaluation=evaluation,
                wait_reasons=waits,
            )
        return decision

    def observation(self, decision: ScheduleDecision) -> PlanningObservation:
        try:
            return self._observations[decision.decision_id]
        except KeyError as error:
            raise ValueError("decision has no Planner observation") from error

    def committed(self, decision: ScheduleDecision) -> None:
        pending = self._pending.get(decision.decision_id)
        if pending is None or pending[0] != decision:
            raise ValueError("decision is not pending")
        self.policy.commit(pending[1], decision_sequence=decision.decision_id)
        del self._pending[decision.decision_id]
        self._observations.pop(decision.decision_id, None)

    def discarded(self, decision: ScheduleDecision) -> None:
        pending = self._pending.get(decision.decision_id)
        if pending is None or pending[0] != decision:
            raise ValueError("decision is not pending")
        del self._pending[decision.decision_id]
        self._observations.pop(decision.decision_id, None)


class PlanningWaitReasonCode(str, Enum):
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INCOMPATIBLE_COHORT = "incompatible_with_selected_cohort"
    RR_STAGE_NOT_SELECTED = "rr_stage_not_selected"


@dataclass(frozen=True, slots=True)
class PlanningWaitReason:
    request_id: str
    stage: SynthesisStage
    reason: PlanningWaitReasonCode
    wait_decisions: int


@dataclass(frozen=True, slots=True)
class PlanningObservation:
    snapshot_sequence: int
    ready: tuple[ReadyStageWork, ...]
    eligible: tuple[ReadyStageWork, ...]
    selected: tuple[ReadyStageWork, ...]
    evaluation: PolicyEvaluation
    wait_reasons: tuple[PlanningWaitReason, ...]


__all__ = [
    "Planner",
    "PlanningObservation",
    "PlanningWaitReason",
    "PlanningWaitReasonCode",
]

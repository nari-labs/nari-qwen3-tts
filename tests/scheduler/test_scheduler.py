from __future__ import annotations

from dataclasses import replace

import pytest

from nari_qwen3_tts.contract import (
    TALKER_DECODE_COMPATIBILITY,
    CodecBatchCompatibility,
    CodecExecutionMode,
    CodePredictorBatchCompatibility,
    ReadyStageWork,
    SynthesisStage,
    TalkerPrefillBatchCompatibility,
)
from nari_qwen3_tts.contract.rng import CodePredictorSamplerRoute
from nari_qwen3_tts.planner import CaptureCatalog, Planner
from nari_qwen3_tts.planner.planner import PlanningWaitReasonCode
from nari_qwen3_tts.planner.policy import RoundRobinPolicy
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader


def _work(
    request_id: str,
    stage: SynthesisStage,
    *,
    compatibility=None,
    admission_sequence: int = 0,
    logical_step: int = 0,
) -> ReadyStageWork:
    if compatibility is None:
        compatibility = {
            SynthesisStage.TALKER_PREFILL: TalkerPrefillBatchCompatibility(10),
            SynthesisStage.CODE_PREDICTOR: CodePredictorBatchCompatibility(),
            SynthesisStage.TALKER_DECODE: TALKER_DECODE_COMPATIBILITY,
            SynthesisStage.CODEC: CodecBatchCompatibility(
                mode=CodecExecutionMode.WARM,
                model_frames=4,
                input_frames=4,
                visible_frames=4,
                pcm_start_frame=0,
                terminal=False,
                producer_frames=4,
            ),
        }[stage]
    return ReadyStageWork(
        request_id=request_id,
        version=0,
        stage=stage,
        logical_step=logical_step,
        compatibility=compatibility,
        admission_sequence=admission_sequence,
        startup=True,
    )


class _PlannerHarness:
    def __init__(self, *, policy=None) -> None:
        config = ProfileLoader().load_profile(ExecutionProfile.TTFA)
        self.planner = Planner(
            catalog=CaptureCatalog.from_config(config.stages),
            policy=policy or RoundRobinPolicy(),
        )
        self.next_decision_id = 1

    def plan(
        self,
        items: tuple[ReadyStageWork, ...],
        *,
        now_s: float = 0.0,
        available_rows: int = 1024,
    ):
        decision = self.planner.plan(
            items,
            now_s=now_s,
            decision_id=self.next_decision_id,
            available_rows=available_rows,
        )
        self.next_decision_id += 1
        return decision


def _capture_size(batch) -> int:
    assert batch.capture is not None
    return batch.capture.key.capture_batch_size


def test_ready_age_is_preserved_only_while_exact_work_remains_ready() -> None:
    harness = _PlannerHarness()
    work = _work("a", SynthesisStage.TALKER_DECODE)
    first = harness.planner._capture((work,))
    second = harness.planner._capture((work,))
    changed = harness.planner._capture((replace(work, logical_step=1),))
    empty = harness.planner._capture(())
    returned = harness.planner._capture((work,))

    assert first[0] < second[0] < changed[0] < empty[0] < returned[0]
    assert first[1][0].ready_sequence == second[1][0].ready_sequence
    assert changed[1][0].ready_sequence > second[1][0].ready_sequence
    assert returned[1][0].ready_sequence > changed[1][0].ready_sequence


def test_planner_releases_observations_when_decisions_leave_pending_state() -> None:
    committed = _PlannerHarness()
    committed_decision = committed.plan((_work("committed", SynthesisStage.TALKER_DECODE),))
    assert committed.planner.observation(committed_decision).selected

    committed.planner.committed(committed_decision)

    assert committed.planner._pending == {}
    assert committed.planner._observations == {}

    discarded = _PlannerHarness()
    discarded_decision = discarded.plan((_work("discarded", SynthesisStage.TALKER_DECODE),))
    assert discarded.planner.observation(discarded_decision).selected

    discarded.planner.discarded(discarded_decision)

    assert discarded.planner._pending == {}
    assert discarded.planner._observations == {}


def test_planner_skips_observation_materialization_when_not_requested() -> None:
    harness = _PlannerHarness()
    decision = harness.planner.plan(
        (_work("request", SynthesisStage.TALKER_DECODE),),
        now_s=0.0,
        decision_id=1,
        available_rows=1,
        record_observation=False,
    )

    assert harness.planner._observations == {}
    with pytest.raises(ValueError, match="no Planner observation"):
        harness.planner.observation(decision)

    harness.planner.committed(decision)
    assert harness.planner._pending == {}


def test_round_robin_is_deterministic_and_independent_of_candidate_order() -> None:
    stages = (
        SynthesisStage.TALKER_PREFILL,
        SynthesisStage.TALKER_DECODE,
        SynthesisStage.CODEC,
        SynthesisStage.CODE_PREDICTOR,
    )

    def sequence(order: tuple[int, ...]) -> tuple[SynthesisStage, ...]:
        harness = _PlannerHarness()
        candidates = tuple(
            _work(f"request-{index}", stage, admission_sequence=index)
            for index, stage in enumerate(stages)
        )
        selected = []
        for _ in range(8):
            decision = harness.plan(tuple(candidates[index] for index in order))
            selected.append(decision.selected_stage)
            harness.planner.committed(decision)
        return tuple(selected)

    expected = sequence((0, 1, 2, 3))
    assert sequence((3, 1, 0, 2)) == expected
    assert expected[:4] == stages
    assert expected[4:] == stages


def test_round_robin_first_codec_precedes_same_request_talker_then_yields() -> None:
    harness = _PlannerHarness()
    first_codec = _work("a", SynthesisStage.CODEC, logical_step=0)
    second_codec = _work("b", SynthesisStage.CODEC, logical_step=0)
    first_talker = _work("a", SynthesisStage.TALKER_DECODE)
    second_talker = _work("b", SynthesisStage.TALKER_DECODE)

    first = harness.plan(
        (first_talker, first_codec, second_talker, second_codec),
        available_rows=1,
    )
    first_observation = harness.planner.observation(first)
    assert first.selected_request_ids == ("a",)
    assert first.selected_stage is SynthesisStage.CODEC
    assert first_observation.evaluation.reason == "first_pass_precedence"
    harness.planner.committed(first)

    yielded = harness.plan(
        (first_talker, second_talker, second_codec),
        available_rows=1,
    )
    assert yielded.selected_stage is SynthesisStage.TALKER_DECODE
    assert harness.planner.observation(yielded).evaluation.reason == RoundRobinPolicy.name


def test_first_pass_trace_retains_precedence_when_filtered_rr_selects_other_stage() -> None:
    harness = _PlannerHarness()
    ordinary_codec = _work("request", SynthesisStage.CODEC, logical_step=1)
    initial = harness.plan((ordinary_codec,))
    harness.planner.committed(initial)

    first_codec = _work("request", SynthesisStage.CODEC, logical_step=0)
    same_request_talker = _work("request", SynthesisStage.TALKER_DECODE)
    cp = _work("cp", SynthesisStage.CODE_PREDICTOR)
    decision = harness.plan((same_request_talker, first_codec, cp))
    evaluation = harness.planner.observation(decision).evaluation

    assert evaluation.rr_counterfactual is SynthesisStage.TALKER_DECODE
    assert decision.selected_stage is SynthesisStage.CODE_PREDICTOR
    assert evaluation.reason == "first_pass_precedence"
    assert evaluation.overrode_round_robin
    harness.planner.committed(decision)

    repeated = harness.plan((same_request_talker, first_codec))
    assert harness.planner.observation(repeated).evaluation.reason == "first_pass_precedence"
    assert repeated.selected_stage is SynthesisStage.CODEC


def test_exact_codec_compatibility_partitions_then_splits_and_pads() -> None:
    compatible = CodecBatchCompatibility(
        CodecExecutionMode.WARM, 4, 4, 4, 0, 4, False
    )
    incompatible = replace(compatible, model_frames=8, input_frames=8, visible_frames=8)
    request_ids = tuple(f"request-{index}" for index in range(18))
    harness = _PlannerHarness()
    candidates = tuple(
        _work(
            request_id,
            SynthesisStage.CODEC,
            compatibility=compatible if index < 17 else incompatible,
            admission_sequence=index,
        )
        for index, request_id in enumerate(request_ids)
    )
    decision = harness.plan(candidates)
    observation = harness.planner.observation(decision)

    assert decision.selected_stage is SynthesisStage.CODEC
    assert [batch.logical_rows for batch in decision.batches] == [8, 8, 1]
    assert [_capture_size(batch) for batch in decision.batches] == [8, 8, 1]
    assert all(batch.compatibility == compatible for batch in decision.batches)
    assert decision.selected_request_ids == request_ids[:17]
    waiting = {wait.request_id: wait.reason for wait in observation.wait_reasons}
    assert waiting[request_ids[-1]] is PlanningWaitReasonCode.INCOMPATIBLE_COHORT


def test_warm_terminal_tails_share_the_same_padded_capture_cohort() -> None:
    short = CodecBatchCompatibility(
        CodecExecutionMode.WARM, 12, 3, 3, 0, 3, True
    )
    longer = replace(short, input_frames=7, visible_frames=7, producer_frames=7)
    harness = _PlannerHarness()
    decision = harness.plan(
        (
            _work("short", SynthesisStage.CODEC, compatibility=short),
            _work("longer", SynthesisStage.CODEC, compatibility=longer),
        )
    )

    assert decision.selected_request_ids == ("longer", "short")
    assert len(decision.batches) == 1
    assert _capture_size(decision.batches[0]) == 2
    assert tuple(row.compatibility for row in decision.batches[0].real_rows) == (longer, short)


def test_whole_sequence_rows_keep_row_local_pcm_windows() -> None:
    context_followup = CodecBatchCompatibility(
        CodecExecutionMode.WHOLE_SEQUENCE, 3, 3, 2, 1, 2, False
    )
    full_window = replace(context_followup, visible_frames=3, pcm_start_frame=0, producer_frames=3)
    harness = _PlannerHarness()
    decision = harness.plan(
        (
            _work("context", SynthesisStage.CODEC, compatibility=context_followup),
            _work("full", SynthesisStage.CODEC, compatibility=full_window),
        )
    )

    assert decision.selected_request_ids == ("context", "full")
    assert len(decision.batches) == 1
    assert _capture_size(decision.batches[0]) == 2


def test_code_predictor_sampler_route_is_exact_batch_compatibility() -> None:
    harness = _PlannerHarness()
    fused = CodePredictorBatchCompatibility(CodePredictorSamplerRoute.FUSED)
    general = CodePredictorBatchCompatibility(CodePredictorSamplerRoute.GENERAL)
    decision = harness.plan(
        (
            _work("fused-a", SynthesisStage.CODE_PREDICTOR, compatibility=fused),
            _work("general", SynthesisStage.CODE_PREDICTOR, compatibility=general),
            _work("fused-b", SynthesisStage.CODE_PREDICTOR, compatibility=fused),
        )
    )

    assert decision.selected_request_ids == ("fused-a", "fused-b")
    assert all(batch.compatibility == fused for batch in decision.batches)
    waits = harness.planner.observation(decision).wait_reasons
    assert {wait.request_id: wait.reason for wait in waits} == {
        "general": PlanningWaitReasonCode.INCOMPATIBLE_COHORT
    }


def test_prefill_planner_keeps_mixed_lengths_and_capture_padding() -> None:
    harness = _PlannerHarness()
    decision = harness.plan(
        tuple(
            _work(
                request_id,
                SynthesisStage.TALKER_PREFILL,
                compatibility=TalkerPrefillBatchCompatibility(length),
            )
            for request_id, length in (("a", 10), ("b", 15), ("c", 11))
        )
    )
    batch = decision.batches[0]

    assert batch.request_ids == ("a", "b", "c")
    assert batch.logical_rows == 3
    assert _capture_size(batch) == 4
    assert batch.padding_rows == 1
    assert tuple(row.compatibility.sequence_length for row in batch.real_rows) == (10, 15, 11)
    assert all(row.padding and row.request_id is None for row in batch.rows[3:])


def test_wait_reasons_distinguish_rr_from_incompatible_selected_stage() -> None:
    harness = _PlannerHarness()
    incompatible_cp = CodePredictorBatchCompatibility()
    candidates = (
        _work("decode", SynthesisStage.TALKER_DECODE),
        _work("cp-a", SynthesisStage.CODE_PREDICTOR, compatibility=incompatible_cp),
        _work("cp-b", SynthesisStage.CODE_PREDICTOR, compatibility=incompatible_cp),
    )
    decision = harness.plan(candidates)
    observation = harness.planner.observation(decision)

    assert decision.selected_stage is SynthesisStage.TALKER_DECODE
    assert {wait.reason for wait in observation.wait_reasons} == {
        PlanningWaitReasonCode.RR_STAGE_NOT_SELECTED
    }
    assert observation.eligible == observation.ready
    assert decision.selected_request_ids == ("decode",)


def test_planner_caps_selection_at_available_claim_capacity() -> None:
    harness = _PlannerHarness()
    candidates = tuple(
        _work(request_id, SynthesisStage.TALKER_PREFILL, admission_sequence=index)
        for index, request_id in enumerate(("a", "b", "c"))
    )
    decision = harness.plan(candidates, available_rows=2)
    waits = harness.planner.observation(decision).wait_reasons

    assert decision.selected_request_ids == ("a", "b")
    assert decision.batches[0].logical_rows == 2
    assert waits[0].request_id == "c"
    assert waits[0].reason is PlanningWaitReasonCode.RESOURCE_EXHAUSTED

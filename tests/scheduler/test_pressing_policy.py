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
from nari_qwen3_tts.planner.planner import PlanningWaitReasonCode
from nari_qwen3_tts.planner.policy import DeadlineAwarePolicy, RoundRobinPolicy

from .test_scheduler import _capture_size, _PlannerHarness


def _work(
    request_id: str,
    stage: SynthesisStage,
    *,
    startup: bool,
    deadline: float | None = None,
    reserve: float = 0.0,
    compatibility=None,
    admission_sequence: int = 0,
) -> ReadyStageWork:
    if compatibility is None:
        compatibility = {
            SynthesisStage.TALKER_PREFILL: TalkerPrefillBatchCompatibility(10),
            SynthesisStage.CODE_PREDICTOR: CodePredictorBatchCompatibility(),
            SynthesisStage.TALKER_DECODE: TALKER_DECODE_COMPATIBILITY,
            SynthesisStage.CODEC: CodecBatchCompatibility(
                CodecExecutionMode.WARM, 12, 12, 12, 0, 12, False
            ),
        }[stage]
    return ReadyStageWork(
        request_id=request_id,
        version=0,
        stage=stage,
        logical_step=0,
        compatibility=compatibility,
        admission_sequence=admission_sequence,
        startup=startup,
        deadline_s=deadline,
        reserve_s=reserve,
    )


def _planner() -> _PlannerHarness:
    return _PlannerHarness(policy=DeadlineAwarePolicy(lead_s=1.0))


def test_urgent_established_codec_precedes_every_startup_tier() -> None:
    items = (
        _work("startup-codec", SynthesisStage.CODEC, startup=True),
        _work("startup-prefill", SynthesisStage.TALKER_PREFILL, startup=True, admission_sequence=1),
        _work(
            "urgent",
            SynthesisStage.CODEC,
            startup=False,
            deadline=5.0,
            reserve=0.2,
            admission_sequence=2,
        ),
    )
    harness = _planner()
    decision = harness.plan(items, now_s=4.8)
    evaluation = harness.planner.observation(decision).evaluation

    assert decision.selected_request_ids[0] == "urgent"
    assert set(decision.selected_request_ids) == {"urgent", "startup-codec"}
    assert evaluation.reason == "deadline_aware_urgent_codec"
    assert evaluation.rr_counterfactual is SynthesisStage.TALKER_PREFILL
    assert evaluation.overrode_round_robin


def test_startup_tiers_follow_codec_prefill_cp_talker_order() -> None:
    remaining = [
        _work("decode", SynthesisStage.TALKER_DECODE, startup=True),
        _work("cp", SynthesisStage.CODE_PREDICTOR, startup=True, admission_sequence=1),
        _work("prefill", SynthesisStage.TALKER_PREFILL, startup=True, admission_sequence=2),
        _work("codec", SynthesisStage.CODEC, startup=True, admission_sequence=3),
    ]
    harness = _planner()
    selected = []
    for now_s in range(4):
        decision = harness.plan(tuple(remaining), now_s=float(now_s))
        selected.append(decision.selected_request_ids[0])
        harness.planner.committed(decision)
        remaining = [work for work in remaining if work.request_id != selected[-1]]

    assert selected == ["codec", "prefill", "cp", "decode"]


def test_startup_anchor_piggybacks_established_rows_of_the_same_action() -> None:
    items = (
        _work("established", SynthesisStage.TALKER_DECODE, startup=False, deadline=20.0),
        _work("starting", SynthesisStage.TALKER_DECODE, startup=True, admission_sequence=1),
    )
    harness = _planner()
    decision = harness.plan(items)

    assert harness.planner.observation(decision).evaluation.reason == "deadline_aware_startup"
    assert decision.selected_request_ids == ("starting", "established")


def test_pressing_anchor_piggybacks_only_exact_compatible_work() -> None:
    compatible = CodecBatchCompatibility(
        CodecExecutionMode.WARM, 12, 12, 12, 0, 12, False
    )
    incompatible = CodecBatchCompatibility(
        CodecExecutionMode.WARM, 8, 8, 8, 0, 8, False
    )
    items = (
        _work("piggyback", SynthesisStage.CODEC, startup=False, deadline=20.0, compatibility=compatible),
        _work("incompatible", SynthesisStage.CODEC, startup=False, deadline=20.0, compatibility=incompatible),
        _work("pressing", SynthesisStage.CODEC, startup=False, deadline=10.0, compatibility=compatible),
    )
    harness = _planner()
    decision = harness.plan(items, now_s=9.2)

    assert harness.planner.observation(decision).evaluation.reason == "deadline_aware"
    assert decision.selected_request_ids[0] == "pressing"
    assert set(decision.selected_request_ids) == {"pressing", "piggyback"}


def test_pressing_cohort_prioritizes_all_pressing_rows_before_nonpressing_work() -> None:
    items = (
        _work("older-nonpressing", SynthesisStage.TALKER_DECODE, startup=False, deadline=20.0),
        _work("pressing-peer", SynthesisStage.TALKER_DECODE, startup=False, deadline=10.1),
        _work("pressing-anchor", SynthesisStage.TALKER_DECODE, startup=False, deadline=10.0),
    )
    harness = _planner()
    decision = harness.plan(items, now_s=9.2, available_rows=2)
    evaluation = harness.planner.observation(decision).evaluation

    assert decision.selected_request_ids == ("pressing-anchor", "pressing-peer")
    assert evaluation.priority_request_ids == ("pressing-anchor", "pressing-peer")


def test_pressing_deadline_tie_uses_stage_order_before_request_id() -> None:
    harness = _planner()
    decision = harness.plan(
        (
            _work("a-cp", SynthesisStage.CODE_PREDICTOR, startup=False, deadline=10.0),
            _work("z-talker", SynthesisStage.TALKER_DECODE, startup=False, deadline=10.0),
        ),
        now_s=9.2,
    )
    evaluation = harness.planner.observation(decision).evaluation

    assert decision.selected_stage is SynthesisStage.TALKER_DECODE
    assert evaluation.anchor_request_id == "z-talker"


def test_pressing_warning_tie_does_not_promote_execution_reserve() -> None:
    harness = _planner()
    decision = harness.plan(
        (
            _work("a", SynthesisStage.CODEC, startup=False, deadline=10.0),
            _work("z", SynthesisStage.CODEC, startup=False, deadline=10.0, reserve=0.5),
        ),
        now_s=9.2,
    )
    evaluation = harness.planner.observation(decision).evaluation

    assert evaluation.reason == "deadline_aware"
    assert evaluation.anchor_request_id == "a"


def test_variable_warm_terminal_pressing_rows_precede_exact_shape_piggyback() -> None:
    long_tail = CodecBatchCompatibility(
        CodecExecutionMode.WARM, 12, 7, 7, 0, 7, True
    )
    short_tail = replace(long_tail, input_frames=3, visible_frames=3, producer_frames=3)
    items = (
        _work("anchor", SynthesisStage.CODEC, startup=False, deadline=10.0, compatibility=long_tail),
        _work(
            "a-piggyback",
            SynthesisStage.CODEC,
            startup=False,
            deadline=20.0,
            compatibility=long_tail,
            admission_sequence=1,
        ),
        _work(
            "z-pressing",
            SynthesisStage.CODEC,
            startup=False,
            deadline=10.1,
            compatibility=short_tail,
            admission_sequence=2,
        ),
    )
    harness = _planner()
    decision = harness.plan(items, now_s=9.2)

    assert harness.planner.observation(decision).evaluation.reason == "deadline_aware"
    assert decision.selected_request_ids == ("anchor", "z-pressing", "a-piggyback")


@pytest.mark.parametrize(
    ("startup", "deadline", "now_s", "reason"),
    (
        (True, None, 0.0, "deadline_aware_startup"),
        (False, 10.0, 9.2, "deadline_aware"),
        (False, 10.0, 10.0, "deadline_aware_urgent_codec"),
    ),
)
def test_deadline_aware_codec_cohort_stays_within_capture_capacity(
    startup: bool,
    deadline: float | None,
    now_s: float,
    reason: str,
) -> None:
    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.COLD, 6, 6, 6, 0, 6, False
    )
    items = tuple(
        _work(
            f"request-{index:02d}",
            SynthesisStage.CODEC,
            startup=startup,
            deadline=deadline,
            reserve=0.25,
            compatibility=compatibility,
            admission_sequence=index,
        )
        for index in range(10)
    )
    harness = _planner()
    decision = harness.plan(items, now_s=now_s)
    observation = harness.planner.observation(decision)

    assert observation.evaluation.reason == reason
    assert len(observation.selected) == 8
    assert len(decision.batches) == 1
    assert _capture_size(decision.batches[0]) == 8
    assert {
        wait.reason
        for wait in observation.wait_reasons
        if wait.stage is SynthesisStage.CODEC
    } == {PlanningWaitReasonCode.RESOURCE_EXHAUSTED}


def test_urgent_codec_cap_uses_completion_deadline_order_for_the_whole_cohort() -> None:
    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.COLD, 6, 6, 6, 0, 6, False
    )
    peers = tuple(
        _work(
            f"peer-{index}",
            SynthesisStage.CODEC,
            startup=False,
            deadline=9.5,
            compatibility=compatibility,
            admission_sequence=index,
        )
        for index in range(8)
    )
    anchor = _work(
        "urgent-anchor",
        SynthesisStage.CODEC,
        startup=False,
        deadline=10.0,
        reserve=1.0,
        compatibility=compatibility,
        admission_sequence=8,
    )
    harness = _planner()
    decision = harness.plan((*peers, anchor), now_s=9.0)
    observation = harness.planner.observation(decision)

    assert observation.evaluation.reason == "deadline_aware_urgent_codec"
    assert observation.evaluation.anchor_request_id == "urgent-anchor"
    assert decision.selected_request_ids == tuple(f"peer-{index}" for index in range(8))
    assert observation.wait_reasons[0].request_id == "urgent-anchor"
    assert observation.wait_reasons[0].reason is PlanningWaitReasonCode.RESOURCE_EXHAUSTED


def test_deadline_aware_empty_terminal_codec_uses_configured_codec_cap() -> None:
    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.EMPTY, 0, 0, 0, 0, 0, True
    )
    items = tuple(
        _work(
            f"request-{index:02d}",
            SynthesisStage.CODEC,
            startup=True,
            compatibility=compatibility,
            admission_sequence=index,
        )
        for index in range(33)
    )
    harness = _planner()
    decision = harness.plan(items)
    observation = harness.planner.observation(decision)

    assert len(observation.selected) == 32
    assert len(decision.batches) == 1
    assert decision.batches[0].capture is None
    assert decision.batches[0].logical_rows == 32
    assert observation.wait_reasons[0].reason is PlanningWaitReasonCode.RESOURCE_EXHAUSTED


@pytest.mark.parametrize(
    ("startup", "deadline", "now_s"),
    ((True, None, 0.0), (False, 10.0, 9.2), (False, 10.0, 10.0)),
)
def test_whole_sequence_phase_uses_followup_cap_even_with_larger_capture(
    startup: bool,
    deadline: float | None,
    now_s: float,
) -> None:
    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.WHOLE_SEQUENCE, 1, 1, 1, 0, 1, False
    )
    items = tuple(
        _work(
            f"request-{index:02d}",
            SynthesisStage.CODEC,
            startup=startup,
            deadline=deadline,
            reserve=0.01,
            compatibility=compatibility,
            admission_sequence=index,
        )
        for index in range(10)
    )
    harness = _planner()
    decision = harness.plan(items, now_s=now_s)

    assert len(harness.planner.observation(decision).selected) == 8
    assert len(decision.batches) == 1
    assert _capture_size(decision.batches[0]) == 8


def test_pressing_falls_back_to_rr_and_discard_grants_no_service_credit() -> None:
    policy = DeadlineAwarePolicy(lead_s=1.0)
    harness = _PlannerHarness(policy=policy)
    decision = harness.plan(
        (
            _work("decode", SynthesisStage.TALKER_DECODE, startup=False, deadline=20.0),
            _work("cp", SynthesisStage.CODE_PREDICTOR, startup=False, deadline=20.0),
        ),
        now_s=1.0,
    )
    evaluation = harness.planner.observation(decision).evaluation
    before = policy.round_robin.checkpoint()
    harness.planner.discarded(decision)

    assert evaluation.reason == RoundRobinPolicy.name
    assert decision.selected_stage is evaluation.rr_counterfactual
    assert policy.round_robin.checkpoint() == before

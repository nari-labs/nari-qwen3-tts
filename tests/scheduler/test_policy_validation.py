from __future__ import annotations

import math
from dataclasses import replace

import pytest

from nari_qwen3_tts.contract import ReadyStageWork, SynthesisStage
from nari_qwen3_tts.engine.pipeline import SynthesisPipeline
from nari_qwen3_tts.planner.policy import DeadlineAwarePolicy, PolicyEvaluation
from nari_qwen3_tts.profile import RequiredSchedulingPolicy, SchedulingPolicyConfig

from .test_pipeline_loop import _runtime
from .test_pressing_policy import _work as _pressing_work
from .test_scheduler import _PlannerHarness, _work


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_ready_work_rejects_non_finite_or_boolean_playback_deadline(value) -> None:
    with pytest.raises((TypeError, ValueError), match="deadline"):
        _pressing_work(
            "request",
            SynthesisStage.TALKER_DECODE,
            startup=False,
            deadline=value,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_ready_work_rejects_non_finite_or_boolean_execution_reserve(value) -> None:
    with pytest.raises((TypeError, ValueError), match="reserve"):
        _pressing_work(
            "request",
            SynthesisStage.CODEC,
            startup=False,
            deadline=10.0,
            reserve=value,
        )


@pytest.mark.parametrize(("startup", "deadline"), [(True, 10.0), (False, None)])
def test_ready_work_rejects_incoherent_startup_and_playback_state(
    startup: bool,
    deadline: float | None,
) -> None:
    with pytest.raises(ValueError, match="startup|deadline|playback"):
        _pressing_work(
            "request",
            SynthesisStage.TALKER_DECODE,
            startup=startup,
            deadline=deadline,
        )


@pytest.mark.parametrize("now_s", [math.nan, math.inf, -math.inf, True])
def test_planner_rejects_non_finite_or_boolean_decision_time(now_s) -> None:
    item = _pressing_work(
        "request",
        SynthesisStage.TALKER_DECODE,
        startup=False,
        deadline=10.0,
    )
    with pytest.raises((TypeError, ValueError), match="time|now"):
        _PlannerHarness().plan((item,), now_s=now_s)


def test_scheduling_policy_config_accepts_zero_pressing_lead() -> None:
    config = SchedulingPolicyConfig(
        kind=RequiredSchedulingPolicy.DEADLINE_AWARE,
        pressing_lead_s=0.0,
    )
    assert config.pressing_lead_s == 0.0


@pytest.mark.parametrize("lead_s", [math.nan, math.inf, -math.inf, True])
def test_scheduling_policy_config_rejects_non_finite_or_boolean_pressing_lead(lead_s) -> None:
    with pytest.raises((TypeError, ValueError), match="lead"):
        SchedulingPolicyConfig(
            kind=RequiredSchedulingPolicy.DEADLINE_AWARE,
            pressing_lead_s=lead_s,
        )


def test_wait_age_is_non_negative_for_every_newly_ready_row() -> None:
    items = tuple(
        _work(request_id, stage, admission_sequence=index)
        for index, (request_id, stage) in enumerate(
            (
                ("prefill", SynthesisStage.TALKER_PREFILL),
                ("decode", SynthesisStage.TALKER_DECODE),
                ("cp", SynthesisStage.CODE_PREDICTOR),
            )
        )
    )
    harness = _PlannerHarness(policy=DeadlineAwarePolicy(lead_s=1.0))
    decision = harness.plan(items)
    waits = harness.planner.observation(decision).wait_reasons
    assert waits
    assert all(wait.wait_decisions >= 0 for wait in waits)


def test_wait_reasons_are_keyed_by_logical_work_not_only_request_id() -> None:
    items = (
        _work("same", SynthesisStage.TALKER_DECODE),
        _work("same", SynthesisStage.CODEC),
    )
    harness = _PlannerHarness(policy=DeadlineAwarePolicy(lead_s=1.0))
    decision = harness.plan(items)
    observation = harness.planner.observation(decision)

    selected = {(work.request_id, work.stage) for work in observation.selected}
    waiting = {(wait.request_id, wait.stage) for wait in observation.wait_reasons}
    assert selected | waiting == {
        ("same", SynthesisStage.TALKER_DECODE),
        ("same", SynthesisStage.CODEC),
    }
    assert selected.isdisjoint(waiting)


def test_ready_age_resets_for_a_new_admission_of_the_same_request_id() -> None:
    harness = _PlannerHarness()
    first_work = _work("request", SynthesisStage.TALKER_DECODE, admission_sequence=1)
    first = harness.planner._capture((first_work,))
    reincarnated = harness.planner._capture((replace(first_work, admission_sequence=2),))
    assert reincarnated[1][0].ready_sequence > first[1][0].ready_sequence


class _MalformedPolicy:
    name = "malformed"

    def __init__(
        self,
        *,
        selected_stage: SynthesisStage,
        anchor_request_id: str | None,
    ) -> None:
        self.selected_stage = selected_stage
        self.anchor_request_id = anchor_request_id

    def choose(
        self,
        eligible: tuple[ReadyStageWork, ...],
        *,
        decision_sequence: int,
        now_s: float,
    ) -> PolicyEvaluation:
        del decision_sequence, now_s
        return PolicyEvaluation(
            self.selected_stage,
            eligible[0].stage,
            self.name,
            anchor_request_id=self.anchor_request_id,
        )

    def commit(self, evaluation: PolicyEvaluation, *, decision_sequence: int) -> None:
        del evaluation, decision_sequence


def test_planner_rejects_policy_stage_that_is_not_eligible() -> None:
    item = _work("request", SynthesisStage.TALKER_DECODE)
    harness = _PlannerHarness(
        policy=_MalformedPolicy(
            selected_stage=SynthesisStage.CODE_PREDICTOR,
            anchor_request_id=None,
        )
    )
    with pytest.raises(ValueError, match="eligible|selected stage"):
        harness.plan((item,))


def test_planner_rejects_policy_anchor_that_is_not_eligible() -> None:
    item = _work("request", SynthesisStage.TALKER_DECODE)
    harness = _PlannerHarness(
        policy=_MalformedPolicy(
            selected_stage=SynthesisStage.TALKER_DECODE,
            anchor_request_id="ghost",
        )
    )
    with pytest.raises(ValueError, match="anchor|eligible"):
        harness.plan((item,))


def test_urgent_codec_tie_uses_latest_safe_start_before_ready_age() -> None:
    items = (
        _pressing_work(
            "late-latest-start",
            SynthesisStage.CODEC,
            startup=False,
            deadline=10.0,
            reserve=0.1,
        ),
        _pressing_work(
            "early-latest-start",
            SynthesisStage.CODEC,
            startup=False,
            deadline=10.0,
            reserve=0.4,
            admission_sequence=10,
        ),
    )
    harness = _PlannerHarness(policy=DeadlineAwarePolicy(lead_s=1.0))
    decision = harness.plan(items, now_s=10.0)
    evaluation = harness.planner.observation(decision).evaluation
    assert evaluation.reason == "deadline_aware_urgent_codec"
    assert evaluation.anchor_request_id == "early-latest-start"


def test_established_warning_does_not_preempt_startup_before_urgent() -> None:
    items = (
        _pressing_work(
            "warning",
            SynthesisStage.TALKER_DECODE,
            startup=False,
            deadline=10.0,
        ),
        _pressing_work(
            "startup",
            SynthesisStage.TALKER_PREFILL,
            startup=True,
            admission_sequence=1,
        ),
    )
    harness = _PlannerHarness(policy=DeadlineAwarePolicy(lead_s=1.0))
    decision = harness.plan(items, now_s=9.5)
    evaluation = harness.planner.observation(decision).evaluation
    assert evaluation.reason == "deadline_aware_startup"
    assert evaluation.anchor_request_id == "startup"


def test_discarded_decision_cannot_retain_pending_policy_credit() -> None:
    item = _work("request", SynthesisStage.TALKER_DECODE)
    harness = _PlannerHarness()
    decision = harness.plan((item,))
    harness.planner.discarded(decision)
    with pytest.raises(ValueError, match="pending"):
        harness.planner.discarded(decision)


def test_injected_policy_must_match_the_profile_required_policy() -> None:
    existing, execution = _runtime()
    with pytest.raises(ValueError, match="policy|round_robin|deadline"):
        SynthesisPipeline(
            executor=execution,
            capture_catalog=existing.catalog,
            policy_config=existing.policy_config,
            model_config=existing.model_config,
            policy=DeadlineAwarePolicy(lead_s=1.0),
        )

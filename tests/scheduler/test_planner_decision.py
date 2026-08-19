from __future__ import annotations

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
from nari_qwen3_tts.planner import CaptureCatalog, Planner
from nari_qwen3_tts.planner.policy import DeadlineAwarePolicy, RoundRobinPolicy
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader


def _work(
    request_id: str,
    stage: SynthesisStage,
    compatibility,
    *,
    admission: int,
    startup: bool,
    deadline_s: float | None = None,
    reserve_s: float = 0.0,
) -> ReadyStageWork:
    return ReadyStageWork(
        request_id=request_id,
        stage=stage,
        version=0,
        logical_step=0,
        compatibility=compatibility,
        admission_sequence=admission,
        startup=startup,
        deadline_s=deadline_s,
        reserve_s=reserve_s,
    )


def _canonical_batch(batch) -> tuple[object, ...]:
    return (
        batch.stage,
        batch.request_ids,
        tuple(row.compatibility for row in batch.rows),
        batch.padding_rows,
        None if batch.capture is None else batch.capture.key,
    )


def _policy(value) -> tuple[object, ...]:
    return (
        value.selected_stage,
        value.rr_counterfactual,
        value.reason,
        value.policy_inputs,
        value.anchor_request_id,
        value.priority_request_ids,
        value.cohort_request_ids,
    )


@pytest.mark.parametrize("policy_name", ("round_robin", "deadline_aware"))
def test_planner_decision_is_deterministic_across_candidate_order(policy_name: str) -> None:
    profile = ExecutionProfile.TTFA if policy_name == "deadline_aware" else ExecutionProfile.BALANCED
    config = ProfileLoader().load_profile(profile)
    catalog = CaptureCatalog.from_config(config.stages)
    if policy_name == "deadline_aware":
        first_policy = DeadlineAwarePolicy(lead_s=1.0)
        second_policy = DeadlineAwarePolicy(lead_s=1.0)
    else:
        first_policy = RoundRobinPolicy()
        second_policy = RoundRobinPolicy()
    first_planner = Planner(catalog=catalog, policy=first_policy)
    second_planner = Planner(catalog=catalog, policy=second_policy)

    startup = policy_name == "round_robin"
    deadline = None if startup else 5.0
    work = (
        _work(
            "prefill",
            SynthesisStage.TALKER_PREFILL,
            TalkerPrefillBatchCompatibility(11),
            admission=0,
            startup=startup,
            deadline_s=deadline,
        ),
        _work(
            "decode",
            SynthesisStage.TALKER_DECODE,
            TALKER_DECODE_COMPATIBILITY,
            admission=1,
            startup=startup,
            deadline_s=deadline,
        ),
        _work(
            "codec-a",
            SynthesisStage.CODEC,
            CodecBatchCompatibility(
                CodecExecutionMode.WARM,
                12,
                12,
                12,
                0,
                12,
                False,
            ),
            admission=2,
            startup=startup,
            deadline_s=deadline,
            reserve_s=0.2,
        ),
        _work(
            "codec-b",
            SynthesisStage.CODEC,
            CodecBatchCompatibility(
                CodecExecutionMode.WARM,
                12,
                12,
                12,
                0,
                12,
                False,
            ),
            admission=3,
            startup=startup,
            deadline_s=None if startup else 5.1,
            reserve_s=0.2,
        ),
        _work(
            "cp",
            SynthesisStage.CODE_PREDICTOR,
            CodePredictorBatchCompatibility(),
            admission=4,
            startup=startup,
            deadline_s=None if startup else 8.0,
        ),
    )

    for decision_id in range(1, 9):
        first = first_planner.plan(
            work,
            now_s=5.0,
            decision_id=decision_id,
            available_rows=64,
        )
        second = second_planner.plan(
            tuple(reversed(work)),
            now_s=5.0,
            decision_id=decision_id,
            available_rows=64,
        )
        first_observation = first_planner.observation(first)
        second_observation = second_planner.observation(second)

        assert tuple(map(_canonical_batch, first.batches)) == tuple(
            map(_canonical_batch, second.batches)
        )
        assert tuple(item.request_id for item in first_observation.selected) == tuple(
            item.request_id for item in second_observation.selected
        )
        assert _policy(first_observation.evaluation) == _policy(
            second_observation.evaluation
        )
        assert tuple(
            (item.request_id, item.stage, item.reason, item.wait_decisions)
            for item in first_observation.wait_reasons
        ) == tuple(
            (item.request_id, item.stage, item.reason, item.wait_decisions)
            for item in second_observation.wait_reasons
        )

        first_planner.committed(first)
        second_planner.committed(second)



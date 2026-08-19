from __future__ import annotations

from nari_qwen3_tts.contract import SynthesisStage
from nari_qwen3_tts.contract.model import SynthesisModelSpec
from nari_qwen3_tts.engine.pipeline import SynthesisPipeline
from nari_qwen3_tts.planner import CaptureCatalog
from nari_qwen3_tts.planner.policy import DeadlineAwarePolicy, RoundRobinPolicy
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader

from .test_pipeline_loop import _FakeExecution, _request


def _runtime(*, trace_enabled: bool) -> SynthesisPipeline:
    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    return SynthesisPipeline(
        executor=_FakeExecution([]),
        capture_catalog=CaptureCatalog.from_config(config.stages),
        policy_config=config.policy,
        model_config=SynthesisModelSpec(
            codec_eos_token_id=31,
            talker_vocab_size=32,
            num_codebooks=16,
            samples_per_frame=2,
            sample_rate=2,
        ),
        max_in_flight_rows=(
            config.stages.talker_decode.max_batch_size
            + config.stages.codec.max_batch_size
        ),
        trace_enabled=trace_enabled,
    )


def test_profile_selects_required_runtime_policy() -> None:
    ttfa_config = ProfileLoader().load_profile(ExecutionProfile.TTFA)
    ttfa = SynthesisPipeline(
        executor=_FakeExecution([]),
        capture_catalog=CaptureCatalog.from_config(ttfa_config.stages),
        policy_config=ttfa_config.policy,
        model_config=SynthesisModelSpec(31, 32, samples_per_frame=2),
    )
    balanced = _runtime(trace_enabled=False)
    throughput_config = ProfileLoader().load_profile(ExecutionProfile.THROUGHPUT)
    throughput = SynthesisPipeline(
        executor=_FakeExecution([]),
        capture_catalog=CaptureCatalog.from_config(throughput_config.stages),
        policy_config=throughput_config.policy,
        model_config=SynthesisModelSpec(31, 32, samples_per_frame=2),
    )

    assert isinstance(ttfa.planner.policy, DeadlineAwarePolicy)
    assert ttfa.planner.policy.lead_s == 1.0
    assert isinstance(balanced.planner.policy, RoundRobinPolicy)
    assert isinstance(throughput.planner.policy, RoundRobinPolicy)


def test_trace_on_off_has_identical_selection_and_request_outputs() -> None:
    def run(enabled: bool):
        runtime = _runtime(trace_enabled=enabled)
        runtime.admit(_request(0))
        runtime.admit(_request(1))
        selected = []
        for step_index in range(4):
            step = runtime.step(now_s=float(step_index) / 10)
            selected.append((step.decision.selected_stage, step.decision.selected_request_ids))
        states = tuple(runtime.request(request_id).committed_view() for request_id in ("request-0", "request-1"))
        return selected, states, runtime.normalized_trace()

    off_selection, off_states, off_trace = run(False)
    on_selection, on_states, on_trace = run(True)

    assert on_selection == off_selection
    assert on_states == off_states
    assert off_trace == ()
    kinds = tuple(event["kind"] for event in on_trace)
    assert kinds.count("decision") == 4
    assert kinds.count("dispatch") == 4
    assert kinds.count("completion") == 4
    assert kinds.count("commit") == 4
    decisions = [event for event in on_trace if event["kind"] == "decision"]
    assert decisions[0]["ready"]
    assert decisions[0]["eligible"]
    assert decisions[0]["selected"]
    assert decisions[0]["row_manifest"]
    assert decisions[0]["rr_counterfactual"] is SynthesisStage.TALKER_PREFILL
    assert "wait_reasons" in decisions[0]
    assert "split_pad" in decisions[0]


def test_failure_trace_records_rejection_without_commit_publication() -> None:
    runtime = _runtime(trace_enabled=True)
    runtime.admit(_request(0))

    def fail(key, call):
        del key, call
        raise RuntimeError("replay failed")

    runtime.executor.talker_prefill_rows = fail
    try:
        runtime.step(now_s=0.0)
    except RuntimeError:
        pass
    events = runtime.normalized_trace()
    assert tuple(event["kind"] for event in events) == (
        "decision",
        "dispatch",
        "completion",
        "commit",
    )
    assert events[-2]["succeeded"] is False
    assert events[-1]["published"] == ()
    assert events[-1]["rejected"] is True

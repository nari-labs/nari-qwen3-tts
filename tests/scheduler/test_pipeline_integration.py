from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nari_qwen3_tts.contract import SynthesisStage
from nari_qwen3_tts.engine.state import GenerationPhase
from nari_qwen3_tts.executor import TalkerExecutionResult
from nari_qwen3_tts.executor.types import TalkerResult

from .test_pipeline_loop import _direct_runtime, _Publication, _request, _runtime


def _drive_to_deferred_decode(runtime) -> None:
    while not (
        runtime.request("request-0").generation.generation_step == 2
        and runtime.request("request-0").generation.phase is GenerationPhase.TALKER_DECODE
    ):
        step = runtime.step(now_s=0.0)
        assert step is not None


def _talker_decode_decision(runtime):
    talker = tuple(
        work
        for work in runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
        if work.stage is SynthesisStage.TALKER_DECODE
    )
    assert talker
    decision = runtime.planner.plan(
        talker,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    prepared = runtime._prepare_execution(decision)
    return prepared, runtime._claim_decision(decision)


def test_async_talker_event_failure_releases_plan_and_request_owner() -> None:
    """event synchronization failure follows the replay-failure cleanup path."""

    class Event:
        @staticmethod
        def query() -> bool:
            return False

        @staticmethod
        def synchronize() -> None:
            raise RuntimeError("event synchronization failed")

    runtime, _execution = _runtime()
    runtime.completion_event_factory = Event
    request = _request(0)
    runtime.admit(replace(request, request=replace(request.request, max_new_tokens=4)))
    _drive_to_deferred_decode(runtime)
    prepared, claim = _talker_decode_decision(runtime)

    runtime.execute_decision(prepared, claim)
    with pytest.raises(RuntimeError, match="event synchronization failed"):
        runtime._drain_host_submissions(block=True)

    state = runtime.request("request-0")
    assert state.generation.claim_token is None
    assert state.generation.claim_batch_id is None
    assert runtime.state_store.in_flight_rows == 0
    assert runtime._host_submission_contexts == {}


def test_delayed_talker_completion_is_returned_by_the_step_that_commits_it() -> None:
    """draining an event cannot make its completion invisible to callers."""

    class Event:
        def __init__(self) -> None:
            self.ready = False

        def query(self) -> bool:
            return self.ready

        def synchronize(self) -> None:
            self.ready = True

    events: list[Event] = []

    def event_factory() -> Event:
        event = Event()
        events.append(event)
        return event

    runtime, _execution = _runtime()
    runtime.completion_event_factory = event_factory
    request = _request(0)
    runtime.admit(replace(request, request=replace(request.request, max_new_tokens=4)))
    _drive_to_deferred_decode(runtime)

    prepared, claim = _talker_decode_decision(runtime)
    deferred = runtime.execute_decision(prepared, claim)

    assert deferred is not None
    assert deferred.completions == ()
    pending_batch_id = next(iter(runtime._host_submission_contexts))
    events[0].ready = True

    observed = runtime.step(now_s=0.1)

    assert observed is not None
    assert pending_batch_id in {
        completion.batch_id for completion in observed.completions
    }


def test_remove_failure_preserves_retryable_runtime_state() -> None:
    """request state and execution resources are removed atomically."""

    runtime, execution = _runtime()
    runtime.admit(_request(0))
    runtime.cancel("request-0")

    def fail_remove(request_id: str) -> None:
        raise RuntimeError(f"resource removal failed for {request_id}")

    execution.remove_request = fail_remove

    with pytest.raises(RuntimeError, match="resource removal failed"):
        runtime.remove("request-0")

    assert runtime.request("request-0").is_removable


def test_duplicate_admission_rollback_does_not_remove_existing_execution_resource() -> None:
    """rollback owns only the resource allocation attempted by that admission."""

    runtime, execution = _runtime()
    resources: set[str] = set()

    def add_request(request_id: str) -> None:
        resources.add(request_id)

    def remove_request(request_id: str) -> None:
        resources.remove(request_id)

    execution.add_request = add_request
    execution.remove_request = remove_request
    runtime.admit(_request(0))

    with pytest.raises(ValueError, match="already admitted"):
        runtime.admit(_request(0))

    assert resources == {"request-0"}
    assert runtime.request("request-0").request_id == "request-0"


def test_exact_boundary_empty_terminal_is_routable_without_starting_playback() -> None:
    """zero-byte terminal metadata crosses the route seam exactly once."""

    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    state = runtime.request("request-0")
    state.codec.output_tracking = True
    state.codec.pending_outputs = 1
    state.codec.compute_terminal = True

    runtime.complete_pcm_output(
        "request-0",
        pcm_bytes=0,
        routed_at_s=None,
        terminal_after=True,
    )

    assert state.codec.playback_started_at_s is None
    assert state.codec.emitted_duration_s == 0.0
    assert state.codec.pending_outputs == 0
    assert state.codec.output_terminal


def test_playback_credit_cannot_exceed_committed_unrouted_pcm() -> None:
    """duplicate callbacks cannot manufacture playback duration."""

    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    state = runtime.request("request-0")
    state.codec.output_tracking = True
    state.codec.pending_outputs = 1

    runtime.complete_pcm_output(
        "request-0",
        pcm_bytes=4,
        routed_at_s=1.0,
        terminal_after=False,
    )
    credited = state.codec.emitted_duration_s

    with pytest.raises(RuntimeError, match="pending PCM output"):
        runtime.complete_pcm_output(
            "request-0",
            pcm_bytes=4,
            routed_at_s=1.1,
            terminal_after=False,
        )

    assert state.codec.emitted_duration_s == credited
    assert state.codec.pending_outputs == 0


def test_playback_route_timestamps_are_monotonic() -> None:
    """a later credit cannot move the causal route clock backwards."""

    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    state = runtime.request("request-0")
    state.codec.output_tracking = True
    state.codec.pending_outputs = 2
    runtime.complete_pcm_output(
        "request-0",
        pcm_bytes=2,
        routed_at_s=2.0,
        terminal_after=False,
    )

    with pytest.raises(ValueError, match="monotonic|route time"):
        runtime.complete_pcm_output(
            "request-0",
            pcm_bytes=2,
            routed_at_s=1.0,
            terminal_after=False,
        )

    assert state.codec.playback_started_at_s == 2.0
    assert state.codec.emitted_duration_s == 0.5


def test_completed_pcm_outputs_release_request_ownership_in_order() -> None:
    """successful delivery cannot leave output ownership for the request lifetime."""

    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    state = runtime.request("request-0")
    state.codec.output_tracking = True
    state.codec.pending_outputs = 2

    runtime.complete_pcm_output(
        "request-0",
        pcm_bytes=4,
        routed_at_s=1.0,
        terminal_after=False,
    )
    assert state.codec.pending_outputs == 1
    runtime.complete_pcm_output(
        "request-0",
        pcm_bytes=2,
        routed_at_s=1.1,
        terminal_after=False,
    )
    assert state.codec.pending_outputs == 0


def test_output_with_extra_physical_rows_fails_before_ticket_construction() -> None:
    """positional unpacking cannot silently ignore executor rows."""

    runtime, _execution = _direct_runtime()
    runtime.admit(_request(0))
    candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    runtime._claim_decision(decision)
    batch = decision.batches[0]
    output = TalkerExecutionResult(
        result=TalkerResult(
            tokens=torch.tensor([4, 5]),
            last_hidden=torch.ones((2, 4)),
            logits=torch.ones((2, 32)),
        ),
        next_seen_token_masks=torch.zeros((2, 32), dtype=torch.bool),
        next_sampling_offsets=torch.tensor([512, 512]),
        kv_publications=(
            _Publication("request-0"),
            _Publication("ghost"),
        ),
    )

    with pytest.raises(ValueError, match="output|row|manifest"):
        runtime._talker_completion(
            batch,
            output,
            (runtime.request("request-0"),),
            decode=False,
        )


def test_route_feedback_changes_the_next_snapshot_from_startup_to_established() -> None:
    """only routed PCM creates the deadline used by pressing policy."""

    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    before = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    assert all(work.startup for work in before)
    assert all(work.deadline_s is None for work in before)
    runtime.mark_pcm_routed("request-0", pcm_bytes=4, routed_at_s=1.0)
    after = runtime.planner.candidates(runtime.state_store.requests, now_s=1.0)

    assert all(not work.startup for work in after)
    assert {work.deadline_s for work in after} == {2.0}

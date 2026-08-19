from __future__ import annotations

import pytest

import nari_qwen3_tts.engine.trace as trace_module
from nari_qwen3_tts.contract import (
    TalkerPrefillBatchCompatibility as TalkerPrefillCompatibility,
)
from nari_qwen3_tts.engine.trace import TraceRecorder

from .test_pipeline_loop import _request, _runtime
from .test_trace import _runtime as _trace_runtime


def test_one_runtime_step_enumerates_ready_candidates_once() -> None:
    """one decision is derived from one committed-state enumeration."""

    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    calls = 0
    original = runtime.planner.candidates

    def counted_ready_candidates(states, *, now_s):
        nonlocal calls
        calls += 1
        return original(states, now_s=now_s)

    runtime.planner.candidates = counted_ready_candidates

    assert runtime.step(now_s=0.0) is not None
    assert calls == 1


def test_trace_off_does_not_format_compatibility_payloads() -> None:
    """disabled observation performs no eager payload construction."""

    runtime, _execution = _runtime()
    runtime.admit(_request(0))

    def forbidden_repr(self) -> str:
        del self
        raise AssertionError("trace-off formatted a compatibility value")

    original = TalkerPrefillCompatibility.__repr__
    TalkerPrefillCompatibility.__repr__ = forbidden_repr
    try:
        assert runtime.step(now_s=0.0) is not None
    finally:
        TalkerPrefillCompatibility.__repr__ = original


def test_trace_record_owns_an_immutable_snapshot_of_mutable_inputs() -> None:
    """later mutation cannot rewrite an earlier causal record."""

    trace = TraceRecorder(enabled=True)
    request_ids = ["a"]
    nested = {"rows": [1]}

    trace.record("decision", request_ids=request_ids, nested=nested)
    request_ids.append("b")
    nested["rows"].append(2)

    event = trace.normalized()[0]
    assert event["request_ids"] == ("a",)
    assert event["nested"] == {"rows": (1,)}


def test_remove_records_cleanup_after_both_state_and_execution_release() -> None:
    """cleanup is a first-class lifecycle event."""

    runtime = _trace_runtime(trace_enabled=True)
    runtime.admit(_request(0))
    runtime.cancel("request-0")

    runtime.remove("request-0")

    cleanup = [event for event in runtime.normalized_trace() if event["kind"] == "cleanup"]
    assert len(cleanup) == 1
    assert cleanup == [
        {
            "sequence": cleanup[0]["sequence"],
            "kind": "cleanup",
            "request_id": "request-0",
            "state_released": True,
            "execution_released": True,
        }
    ]


def test_pcm_route_feedback_is_visible_in_the_causal_trace() -> None:
    """pressing inputs must be attributable to successful output routing."""

    runtime = _trace_runtime(trace_enabled=True)
    runtime.admit(_request(0))

    runtime.mark_pcm_routed("request-0", pcm_bytes=4, routed_at_s=1.25)

    routed = [event for event in runtime.normalized_trace() if event["kind"] == "output_routed"]
    assert len(routed) == 1
    assert routed == [
        {
            "sequence": routed[0]["sequence"],
            "kind": "output_routed",
            "request_id": "request-0",
            "pcm_bytes": 4,
            "routed_at_s": 1.25,
            "playback_started_at_s": 1.25,
            "emitted_duration_s": 1.0,
        }
    ]


def test_decision_trace_uses_complete_structured_row_records() -> None:
    """traces reconstruct identity and pressing inputs without repr parsing."""

    runtime = _trace_runtime(trace_enabled=True)
    runtime.admit(_request(0))

    runtime.step(now_s=0.0)

    decision = next(
        event for event in runtime.normalized_trace() if event["kind"] == "decision"
    )
    required_work_fields = {
        "request_id",
        "version",
        "lane",
        "stage",
        "logical_step",
        "admission_sequence",
        "ready_sequence",
        "startup",
        "deadline_s",
        "reserve_s",
        "compatibility",
    }
    assert decision["ready"]
    assert all(
        isinstance(work, dict) and required_work_fields <= set(work)
        for work in decision["ready"]
    )
    assert "compatibility_partition" in decision
    required_row_fields = {
        "physical_row",
        "request_id",
        "version",
        "lane",
        "stage",
        "logical_step",
        "compatibility",
        "padding",
    }
    assert all(
        isinstance(row, dict) and required_row_fields <= set(row)
        for batch in decision["row_manifest"]
        for row in batch
    )


def test_enabled_trace_storage_is_explicitly_bounded() -> None:
    """observation memory cannot grow with total request duration."""

    trace = TraceRecorder(enabled=True, max_events=2)
    trace.record("first")
    trace.record("second")
    trace.record("third")

    assert [event["kind"] for event in trace.normalized()] == ["second", "third"]


def test_trace_hot_path_uses_a_preallocated_overwrite_ring() -> None:
    """wraparound is O(1), without append/delete list churn."""

    trace = TraceRecorder(enabled=True, max_events=2)
    payload_ring = trace._payload_ring
    sequence_ring = trace._sequence_ring
    payload_ring_id = id(payload_ring)
    sequence_ring_id = id(sequence_ring)

    for index in range(10):
        trace.record("event", index=index)

    assert id(trace._payload_ring) == payload_ring_id
    assert id(trace._sequence_ring) == sequence_ring_id
    assert len(payload_ring) == 2
    assert len(sequence_ring) == 2
    assert [event["index"] for event in trace.normalized()] == [8, 9]


def test_trace_hot_path_does_not_materialize_public_events() -> None:
    """TraceEvent construction belongs to after-drain observation only."""

    trace = TraceRecorder(enabled=True)
    original = trace_module.TraceEvent

    def forbidden_trace_event(*args, **kwargs):
        del args, kwargs
        raise AssertionError("hot path materialized a public TraceEvent")

    trace_module.TraceEvent = forbidden_trace_event
    try:
        trace.record("event", request_id="request-0")
    finally:
        trace_module.TraceEvent = original

    assert trace.normalized()[0]["request_id"] == "request-0"


def test_runtime_trace_ring_retains_only_compact_immutable_payloads() -> None:
    """runtime decisions defer nested dict/list expansion until drain."""

    runtime = _trace_runtime(trace_enabled=True)
    runtime.admit(_request(0))
    runtime.step(now_s=0.0)

    def mutable_container_found(value: object) -> bool:
        if isinstance(value, (dict, list)):
            return True
        if isinstance(value, tuple):
            return any(mutable_container_found(item) for item in value)
        return False

    retained_payloads = tuple(
        payload
        for sequence, payload in zip(
            runtime.trace._sequence_ring,
            runtime.trace._payload_ring,
            strict=True,
        )
        if sequence
    )
    assert retained_payloads
    assert not any(mutable_container_found(payload) for payload in retained_payloads)


def test_gc_enabled_trace_ring_retains_only_serialized_payloads(monkeypatch) -> None:
    """GC-safe mode cannot retain a tracked cyclic object structure per event."""

    monkeypatch.setattr(trace_module.gc, "isenabled", lambda: True)
    trace = TraceRecorder(enabled=True, max_events=2)
    trace.record("event", nested={"rows": [1]})

    retained = [
        payload
        for sequence, payload in zip(
            trace._sequence_ring,
            trace._payload_ring,
            strict=True,
        )
        if sequence
    ]
    assert retained and all(isinstance(payload, bytes) for payload in retained)
    assert trace.normalized()[0]["nested"] == {"rows": (1,)}


def test_direct_object_trace_requires_cyclic_gc_to_be_disabled(monkeypatch) -> None:
    """the faster object ring is fail-closed under automatic cyclic GC."""

    monkeypatch.setattr(trace_module.gc, "isenabled", lambda: True)
    with pytest.raises(RuntimeError, match="cyclic GC"):
        TraceRecorder(enabled=True, direct_object_ring=True)

    monkeypatch.setattr(trace_module.gc, "isenabled", lambda: False)
    trace = TraceRecorder(enabled=True, direct_object_ring=True)
    trace.record("event", value=1)
    assert isinstance(trace._payload_ring[0], tuple)
    assert trace.normalized()[0]["value"] == 1


def test_timestamped_direct_trace_records_the_hot_path_clock(monkeypatch) -> None:
    """Diagnostic runs retain causal time without allocating public events."""

    monkeypatch.setattr(trace_module.gc, "isenabled", lambda: False)
    monkeypatch.setattr(trace_module.time, "perf_counter_ns", lambda: 123_456_789)
    trace = TraceRecorder(
        enabled=True,
        direct_object_ring=True,
        timestamps=True,
    )

    trace.record("event", value=1)

    assert trace.normalized()[0]["monotonic_ns"] == 123_456_789

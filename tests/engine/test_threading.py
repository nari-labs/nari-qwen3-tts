from __future__ import annotations

import threading
from collections import defaultdict

import pytest
from server.fakes import FakeEngine, FakeEngineBackend, engine_config, request

from nari_qwen3_tts.contract.errors import ServiceUnavailable
from nari_qwen3_tts.engine.engine import Engine


class _ThreadAuditPort(FakeEngineBackend):
    """Record every engine-facing operation without changing its fake semantics."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int, str | None]] = []
        self.advance_user_requests = False

    def _record(self, name: str, request_id: str | None = None) -> None:
        self.calls.append((name, threading.get_ident(), request_id))

    def execution_context(self):
        parent = super().execution_context()
        audit = self

        class _Context:
            def __enter__(self):
                value = parent.__enter__()
                audit._record("owner_context_enter")
                return value

            def __exit__(self, exc_type, exc_value, traceback):
                audit._record("owner_context_exit")
                return parent.__exit__(exc_type, exc_value, traceback)

        return _Context()

    def health(self):
        self._record("health")
        return super().health()

    def admit(self, request_id, request, *, admitted_at_s, live=False):
        self._record("admit", request_id)
        return super().admit(
            request_id,
            request,
            admitted_at_s=admitted_at_s,
            live=live,
        )

    def tokenize_fragment(self, text, *, is_initial, is_final):
        self._record("tokenize_fragment")
        return super().tokenize_fragment(text, is_initial=is_initial, is_final=is_final)

    def update_request_input(self, request_id, token_ids, *, sequence, is_final):
        self._record("update_request_input", request_id)
        return super().update_request_input(
            request_id,
            token_ids,
            sequence=sequence,
            is_final=is_final,
        )

    def step(self, *, now_s):
        self._record("step")
        if self.advance_user_requests:
            return super().step(now_s=now_s)
        for request_id, state in self.requests.items():
            if not request_id.startswith("readiness-"):
                continue
            if state.cancelled or state.cursor >= len(state.chunks):
                continue
            state.cursor += 1
            for name in self.stage_counts:
                self.stage_counts[name] += 1
            return object()
        return None

    def snapshot(self, request_id, *, pcm_start_index=0):
        self._record("snapshot", request_id)
        return super().snapshot(request_id, pcm_start_index=pcm_start_index)

    def record_pcm_routed(self, request_id, *, pcm_bytes, routed_at_s):
        self._record("record_pcm_routed", request_id)
        return super().record_pcm_routed(
            request_id,
            pcm_bytes=pcm_bytes,
            routed_at_s=routed_at_s,
        )

    def cancel(self, request_id):
        self._record("cancel", request_id)
        return super().cancel(request_id)

    def remove(self, request_id):
        self._record("remove", request_id)
        return super().remove(request_id)


def _service(port: FakeEngineBackend) -> Engine:
    return FakeEngine(
        port,
        config=engine_config(
            max_active_requests=4,
            max_buffered_bytes=64,
            owner_poll_interval_s=0.0005,
        ),
    )


def test_submit_append_cancel_and_pcm_poll_stay_on_the_engine_thread() -> None:
    caller_thread = threading.get_ident()
    port = _ThreadAuditPort()
    service = _service(port)
    service.start(request("probe"), timeout_s=2)
    try:
        stream = service.begin_live(
            "live",
            request("ab "),
            initial_token_ids=(11,),
            initial_wrapped_ids=(101, 102, 103, 11, 104),
            input_finished=False,
            timeout_s=1,
        )
        assert service.append_text("live", (12,), sequence=0, is_final=False, timeout_s=1) is None
        assert service.append_text("live", (13,), sequence=1, is_final=True, timeout_s=1) is None
        service.cancel("live", timeout_s=1)
        with pytest.raises(Exception, match="cancel"):
            stream.read(timeout_s=1)
        assert service.wait_idle(timeout_s=1)
    finally:
        service.stop(timeout_s=2)

    by_name: dict[str, set[int]] = defaultdict(set)
    for name, thread_id, _request_id in port.calls:
        by_name[name].add(thread_id)

    expected_operations = {
        "owner_context_enter",
        "health",
        "admit",
        "update_request_input",
        "step",
        "snapshot",
        "cancel",
        "remove",
        "owner_context_exit",
    }
    assert expected_operations <= by_name.keys()
    engine_threads = {thread_id for _name, thread_id, _request_id in port.calls}
    assert engine_threads == {service.engine_thread_id}
    assert caller_thread not in engine_threads


def test_each_engine_turn_steps_before_polling_pcm() -> None:
    port = _ThreadAuditPort()
    port.advance_user_requests = True
    service = _service(port)
    service.start(request("probe"), timeout_s=2)
    try:
        port.calls.clear()
        stream = service.submit("ordered", request("xy"), timeout_s=1)
        assert b"".join(stream.iter_bytes(timeout_s=1)) == b"xyxy"
        assert service.wait_idle(timeout_s=1)
    finally:
        service.stop(timeout_s=2)

    request_events = [
        name
        for name, _thread_id, request_id in port.calls
        if request_id == "ordered" or name == "step"
    ]
    first_snapshot = request_events.index("snapshot")
    assert "step" in request_events[:first_snapshot]


def test_stop_reclaims_active_requests_and_rejects_later_commands() -> None:
    port = _ThreadAuditPort()
    service = _service(port)
    service.start(request("probe"), timeout_s=2)
    stream = service.submit("active", request("xy"), timeout_s=1)

    service.stop(timeout_s=2)

    assert port.requests == {}
    assert "active" in port.removed
    assert not service._records
    assert service._thread is not None and not service._thread.is_alive()
    with pytest.raises(ServiceUnavailable):
        service.submit("late", request("x"), timeout_s=0.01)
    with pytest.raises(ServiceUnavailable):
        service.begin_live(
            "late-live",
            request("x"),
            initial_token_ids=(11,),
            initial_wrapped_ids=(101, 102, 103, 11, 104),
            input_finished=False,
            timeout_s=0.01,
        )
    with pytest.raises(ServiceUnavailable):
        service.append_text("active", (11,), sequence=0, is_final=True, timeout_s=0.01)
    with pytest.raises(ServiceUnavailable):
        service.cancel("active", timeout_s=0.01)
    with pytest.raises(Exception, match="stopping|cancel"):
        stream.read(timeout_s=1)


def test_startup_failure_leaves_no_request_or_engine_thread() -> None:
    class _FailedHealthPort(_ThreadAuditPort):
        def health(self):
            value = super().health()
            return type(value)(
                required_keys=value.required_keys,
                captured_keys=value.captured_keys - 1,
                required_cuda_graph_instances=value.required_cuda_graph_instances,
                captured_cuda_graph_instances=value.captured_cuda_graph_instances,
                capture_failures=value.capture_failures,
                eager_fallbacks=value.eager_fallbacks,
                stages=value.stages,
            )

    port = _FailedHealthPort()
    service = _service(port)

    with pytest.raises(ServiceUnavailable, match="startup failed"):
        service.start(request("probe"), timeout_s=2)

    assert port.requests == {}
    assert not service._records
    assert service._thread is not None and not service._thread.is_alive()
    assert not service.readiness().ready


def test_engine_step_delegates_without_a_separate_ready_materialization() -> None:
    engine = object.__new__(Engine)
    sentinel = object()

    class _Pipeline:
        @staticmethod
        def has_ready_work() -> bool:
            raise AssertionError("Engine must not materialize ready work before Pipeline.step")

        @staticmethod
        def step(*, now_s: float):
            assert now_s == 1.25
            return sentinel

    engine.pipeline = _Pipeline()

    assert engine._step_execution(now_s=1.25) is sentinel

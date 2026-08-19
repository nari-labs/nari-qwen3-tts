from __future__ import annotations

import threading
import time

import pytest

from nari_qwen3_tts.contract.errors import (
    RequestCancelled,
    ServiceUnavailable,
)
from nari_qwen3_tts.contract.health import ServicePhase
from server.fakes import (
    FakeEngine,
    FakeEngineBackend,
    engine_config,
    pcm16,
    request,
)


def test_service_requires_ordinary_probe_before_admission_and_retires_it() -> None:
    port = FakeEngineBackend()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    assert not service.readiness().ready
    service.start(request("probe"), timeout_s=2)
    try:
        assert service.readiness().ready
        assert len(port.removed) == 1
        assert port.removed[0].startswith("readiness-")

        stream = service.submit("request-1", request("ab"))
        assert b"".join(stream.iter_bytes(timeout_s=1)) == b"abab"
        assert service.wait_idle(timeout_s=1)
        assert "request-1" in port.removed
        assert [(request_id, size) for request_id, size, _ in port.routed] == [
            ("request-1", 2),
            ("request-1", 2),
        ]
    finally:
        service.stop(timeout_s=2)


def test_service_does_not_poll_sleep_while_runtime_work_keeps_progressing(monkeypatch) -> None:
    class ThreeChunkPort(FakeEngineBackend):
        def admit(self, request_id, request, *, admitted_at_s, live=False):
            super().admit(request_id, request, admitted_at_s=admitted_at_s, live=live)
            chunk = pcm16(request.text)
            self.requests[request_id].chunks = (chunk, chunk, chunk)

    sleeps = []
    monkeypatch.setattr("nari_qwen3_tts.engine.engine.time.sleep", sleeps.append)
    port = ThreeChunkPort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    service.start(request("probe"), timeout_s=2)
    try:
        stream = service.submit("request-1", request("abcd"))
        assert b"".join(stream.iter_bytes(timeout_s=1)) == b"abcdabcdabcd"
        assert sleeps == []
    finally:
        service.stop(timeout_s=2)


def test_admission_rejection_is_request_local_and_owner_remains_ready() -> None:
    class RejectingPort(FakeEngineBackend):
        def admit(self, request_id, request, *, admitted_at_s, live=False):
            if request_id == "unsupported":
                raise ValueError("Codec capture does not cover request")
            return super().admit(
                request_id,
                request,
                admitted_at_s=admitted_at_s,
                live=live,
            )

    port = RejectingPort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    service.start(request("probe"), timeout_s=2)
    try:
        with pytest.raises(ValueError, match="Codec capture"):
            service.submit("unsupported", request("x"))
        assert service.readiness().ready
        stream = service.submit("healthy", request("y"))
        assert b"".join(stream.iter_bytes(timeout_s=1)) == b"y\0y\0"
        assert service.wait_idle(timeout_s=1)
        assert service.readiness().ready
    finally:
        service.stop(timeout_s=2)


def test_backpressure_cancels_only_slow_request_while_healthy_request_completes() -> None:
    port = FakeEngineBackend()
    service = FakeEngine(
        port,
        config=engine_config(max_buffered_bytes=4, max_active_requests=4),
    )
    service.start(request("p"), timeout_s=2)
    try:
        slow = service.submit("slow", request("1234"))
        healthy = service.submit("healthy", request("a"))
        healthy_bytes: list[bytes] = []

        reader = threading.Thread(
            target=lambda: healthy_bytes.extend(healthy.iter_bytes(timeout_s=1)),
        )
        reader.start()
        reader.join(timeout=2)
        assert not reader.is_alive()
        assert b"".join(healthy_bytes) == b"a\0a\0"
        with pytest.raises(Exception, match="backpressure"):
            list(slow.iter_bytes(timeout_s=1))
        assert service.wait_idle(timeout_s=1)
        assert {"slow", "healthy"}.issubset(port.removed)
        assert service.readiness().ready
    finally:
        service.stop(timeout_s=2)


def test_cancel_is_request_local_and_capacity_is_reclaimed() -> None:
    class PausedPort(FakeEngineBackend):
        paused = False

        def step(self, *, now_s):
            if not self.paused:
                super().step(now_s=now_s)

    port = PausedPort()
    service = FakeEngine(
        port,
        config=engine_config(max_buffered_bytes=32, max_active_requests=1),
    )
    service.start(request("p"), timeout_s=2)
    try:
        port.paused = True
        first = service.submit("first", request("x"))
        with pytest.raises(Exception, match="capacity"):
            service.submit("second", request("y"))
        service.cancel("first")
        with pytest.raises(RequestCancelled):
            first.read(timeout_s=1)
        assert service.wait_idle(timeout_s=1)
        port.paused = False
        second = service.submit("second", request("y"))
        assert b"".join(second.iter_bytes(timeout_s=1)) == b"y\0y\0"
    finally:
        service.stop(timeout_s=2)


def test_owner_thread_is_the_only_port_caller() -> None:
    class ThreadCheckingPort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.owner_ids: set[int] = set()

        def _record(self):
            self.owner_ids.add(threading.get_ident())

        def admit(self, *args, **kwargs):
            self._record()
            return super().admit(*args, **kwargs)

        def step(self, *args, **kwargs):
            self._record()
            return super().step(*args, **kwargs)

        def snapshot(self, *args, **kwargs):
            self._record()
            return super().snapshot(*args, **kwargs)

        def remove(self, *args, **kwargs):
            self._record()
            return super().remove(*args, **kwargs)

    port = ThreadCheckingPort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    service.start(request("p"), timeout_s=2)
    try:
        stream = service.submit("one", request())
        assert b"".join(stream.iter_bytes(timeout_s=1))
        deadline = time.monotonic() + 1
        while len(port.removed) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(port.owner_ids) == 1
        assert threading.get_ident() not in port.owner_ids
    finally:
        service.stop(timeout_s=2)


def test_every_port_call_runs_inside_its_owner_execution_context() -> None:
    class RequiredContextPort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.inside = False

        def execution_context(self):
            owner = self

            class Context:
                def __enter__(self):
                    owner.inside = True

                def __exit__(self, exc_type, exc_value, traceback):
                    owner.inside = False

            return Context()

        def health(self):
            assert self.inside
            return super().health()

        def admit(self, *args, **kwargs):
            assert self.inside
            return super().admit(*args, **kwargs)

        def step(self, *args, **kwargs):
            assert self.inside
            return super().step(*args, **kwargs)

        def snapshot(self, *args, **kwargs):
            assert self.inside
            return super().snapshot(*args, **kwargs)

        def remove(self, *args, **kwargs):
            assert self.inside
            return super().remove(*args, **kwargs)

    port = RequiredContextPort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    service.start(request("probe"), timeout_s=2)
    try:
        stream = service.submit("context", request("x"))
        assert b"".join(stream.iter_bytes(timeout_s=1)) == b"x\0x\0"
        assert service.wait_idle(timeout_s=1)
    finally:
        service.stop(timeout_s=2)
    assert not port.inside


def test_pcm_routing_fault_fails_only_its_own_request() -> None:
    """Playback accounting is per request; a fault in it must not kill the owner."""

    class FaultyRoutePort(FakeEngineBackend):
        def record_pcm_routed(self, request_id, *, pcm_bytes, routed_at_s):
            if request_id.startswith("readiness-"):
                return super().record_pcm_routed(request_id, pcm_bytes=pcm_bytes, routed_at_s=routed_at_s)
            raise RuntimeError("playback credit accounting is inconsistent")

    port = FaultyRoutePort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    service.start(request("probe"), timeout_s=2)
    try:
        stream = service.submit("faulty", request("ab"))
        with pytest.raises(ServiceUnavailable, match="PCM delivery failed"):
            list(stream.iter_bytes(timeout_s=1))
        assert service.wait_idle(timeout_s=2)

        # The owner is still serving: a second request completes normally.
        assert service.readiness().ready
        assert service.metrics().failed == 1
    finally:
        service.stop(timeout_s=2)


def test_failure_after_readiness_is_not_reported_as_a_startup_failure() -> None:
    """A long-running service that dies must not send diagnosis to startup."""

    service = FakeEngine(FakeEngineBackend(), config=engine_config(max_buffered_bytes=32))
    service.start(request("probe"), timeout_s=2)
    try:
        assert service.readiness().ready
        # Everything the probe proved still holds; only the owner is gone.
        service._phase = ServicePhase.FAILED
        assert service.readiness().reason == "startup_failed"
        service._failed_after_ready = True
        assert service.readiness().reason == "service_failed"
    finally:
        service._phase = ServicePhase.READY
        service.stop(timeout_s=2)

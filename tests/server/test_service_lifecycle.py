from __future__ import annotations

import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

from nari_qwen3_tts.config import EngineConfig, LiveInputConfig
from nari_qwen3_tts.engine.engine import Engine
from server.fakes import (
    FakeEngine,
    FakeEngineBackend,
    FakeRequestState,
    engine_config,
    request,
)


def _cancel_if_present(service: Engine, request_id: str) -> None:
    try:
        service.cancel(request_id, timeout_s=0.5)
    except (KeyError, FutureTimeoutError):
        pass


def test_exact_boundary_empty_terminal_is_not_routed_as_playback() -> None:
    """zero-byte terminal metadata is visible but non-crediting."""

    class EmptyTerminalPort(FakeEngineBackend):
        def admit(self, request_id, request, *, admitted_at_s, live=False):
            if request_id.startswith("readiness-"):
                return super().admit(
                    request_id,
                    request,
                    admitted_at_s=admitted_at_s,
                    live=live,
                )
            del request, admitted_at_s, live
            self.requests[request_id] = FakeRequestState((b"",))

    port = EmptyTerminalPort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    service.start(request("probe"), timeout_s=2)
    try:
        stream = service.submit("empty-terminal", request("x"))
        assert list(stream.iter_bytes(timeout_s=1)) == [b""]
        assert service.wait_idle(timeout_s=1)
        assert not [entry for entry in port.routed if entry[0] == "empty-terminal"]
        assert service.readiness().ready
    finally:
        service.stop(timeout_s=2)


def test_timed_out_submit_cannot_execute_later_as_an_orphan() -> None:
    """a caller timeout tombstones the queued/running admission."""

    class BlockingAdmissionPort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.paused: set[str] = set()

        def admit(self, request_id, request, *, admitted_at_s, live=False):
            if request_id == "timed-out":
                self.entered.set()
                self.release.wait(timeout=2)
                self.paused.add(request_id)
            return super().admit(
                request_id,
                request,
                admitted_at_s=admitted_at_s,
                live=live,
            )

        def step(self, *, now_s):
            for request_id in self.paused:
                request = self.requests.get(request_id)
                if request is not None:
                    request.input_finished = False
            return super().step(now_s=now_s)

    port = BlockingAdmissionPort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    service.start(request("probe"), timeout_s=2)
    try:
        with pytest.raises(FutureTimeoutError):
            service.submit("timed-out", request("x"), timeout_s=0.02)
        assert port.entered.wait(timeout=1)
        port.release.set()

        assert service.wait_idle(timeout_s=0.5)
        assert "timed-out" not in port.requests
        assert service.metrics().active == 0
    finally:
        port.release.set()
        _cancel_if_present(service, "timed-out")
        service.stop(timeout_s=2)


def test_startup_timeout_leaves_a_stoppable_probe_and_no_engine_owner() -> None:
    """startup timeout cannot strand the owner inside the probe."""

    class StalledProbePort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()

        def step(self, *, now_s):
            if not self.release.is_set():
                return None
            return super().step(now_s=now_s)

    port = StalledProbePort()
    service = FakeEngine(port, config=engine_config(owner_poll_interval_s=0.001))
    try:
        with pytest.raises(TimeoutError, match="readiness"):
            service.start(request("probe"), timeout_s=0.02)

        service.stop(timeout_s=0.1)
        assert service.wait_idle(timeout_s=0.1)
        assert port.requests == {}
    finally:
        port.release.set()
        try:
            service.stop(timeout_s=2)
        except (FutureTimeoutError, TimeoutError):
            pass


def test_service_config_owns_all_live_input_and_command_turn_bounds() -> None:
    """flood and update limits are explicit configuration."""

    config = EngineConfig(
        max_commands_per_turn=2,
        live_input=LiveInputConfig(
            max_input_append_events=8,
            max_live_text_tokens=32,
            max_update_tokens=4,
        ),
    )

    assert config.live_input.max_input_append_events == 8
    assert config.live_input.max_live_text_tokens == 32
    assert config.live_input.max_update_tokens == 4
    assert config.max_commands_per_turn == 2


def test_duplicate_cancel_is_one_engine_cancel_and_one_metric_transition() -> None:
    """cancellation ownership is idempotent under client races."""

    class PausedCancelPort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_calls: list[str] = []
            self.allow_removal = False

        def step(self, *, now_s):
            del now_s

        def cancel(self, request_id):
            self.cancel_calls.append(request_id)
            self.requests[request_id].cancelled = self.allow_removal

    port = PausedCancelPort()
    service = FakeEngine(port, config=engine_config(max_buffered_bytes=32))
    # Let only the ordinary probe progress, then pause user work.
    original_step = port.step
    port.step = FakeEngineBackend.step.__get__(port, PausedCancelPort)
    service.start(request("probe"), timeout_s=2)
    port.step = original_step
    try:
        service.submit("cancel-race", request("x"))
        service.cancel("cancel-race")
        service.cancel("cancel-race")

        assert port.cancel_calls == ["cancel-race"]
        assert service.metrics().cancelled == 1
    finally:
        port.allow_removal = True
        if "cancel-race" in port.requests:
            port.requests["cancel-race"].cancelled = True
        port.step = FakeEngineBackend.step.__get__(port, PausedCancelPort)
        service.wait_idle(timeout_s=1)
        service.stop(timeout_s=2)

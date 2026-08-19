from __future__ import annotations

import queue
import threading
from concurrent.futures import Future

import pytest
import torch
from server.fakes import FakeEngine, FakeEngineBackend, engine_config, request

from nari_qwen3_tts.contract.command import (
    AppendText,
    CancelRequest,
    StopEngine,
    SubmitRequest,
)


def test_engine_queue_contains_only_canonical_commands() -> None:
    engine = FakeEngine(FakeEngineBackend(), config=engine_config(max_buffered_bytes=32))
    command_types: list[type[object]] = []
    put = engine._put

    def audit_put(command, *, timeout_s: float) -> None:
        command_types.append(type(command))
        put(command, timeout_s=timeout_s)

    engine._put = audit_put
    engine.start(request("probe"), timeout_s=2)
    try:
        stream = engine.begin_live(
            "live",
            request("ab "),
            initial_token_ids=(11,),
            initial_wrapped_ids=(101, 102, 103, 11, 104),
            input_finished=False,
            timeout_s=1,
        )
        assert engine.append_text("live", (12,), sequence=0, is_final=False, timeout_s=1) is None
        assert engine.append_text("live", (13,), sequence=1, is_final=True, timeout_s=1) is None
        with pytest.raises(KeyError, match="unknown request"):
            engine.cancel("already-retired", timeout_s=1)
        try:
            stream.read(timeout_s=1)
        except Exception:
            pass
    finally:
        engine.stop(timeout_s=2)

    assert set(command_types) == {SubmitRequest, AppendText, CancelRequest, StopEngine}


def test_public_engine_calls_only_enqueue_and_wait_on_the_caller_thread() -> None:
    engine = FakeEngine(FakeEngineBackend())
    caller = threading.get_ident()
    called: list[tuple[type[object], int]] = []

    def put(command, *, timeout_s: float) -> None:
        del timeout_s
        called.append((type(command), threading.get_ident()))
        if isinstance(command, SubmitRequest):
            command.reply.set_result(object())
        elif isinstance(command, AppendText):
            command.reply.set_result(0)
        else:
            command.reply.set_result(None)

    engine._require_ready = lambda: None
    engine._put = put

    assert engine.submit("r", request("x"), timeout_s=0.1) is not None
    assert engine.append_text("r", (1,), sequence=0, is_final=False, timeout_s=0.1) is None
    engine.cancel("r", timeout_s=0.1)

    assert called == [
        (SubmitRequest, caller),
        (AppendText, caller),
        (CancelRequest, caller),
    ]


def test_engine_command_queue_is_bounded_by_config() -> None:
    engine = FakeEngine(FakeEngineBackend(), config=engine_config(command_capacity=1))

    assert isinstance(engine._commands, queue.Queue)
    assert engine._commands.maxsize == 1
    engine._commands.put_nowait(CancelRequest("one", Future()))
    assert engine._commands.full()


def test_engine_thread_enforces_affinity_and_inference_mode() -> None:
    class InferenceAuditPort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.inference_modes: list[bool] = []

        def health(self):
            self.inference_modes.append(torch.is_inference_mode_enabled())
            return super().health()

    port = InferenceAuditPort()
    engine = FakeEngine(port)
    engine.start(request("probe"), timeout_s=2)
    try:
        with pytest.raises(RuntimeError, match="Engine thread"):
            engine._poll_records()
    finally:
        engine.stop(timeout_s=2)

    assert port.inference_modes
    assert all(port.inference_modes)

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import replace

import pytest
import torch
from scheduler.test_pipeline_loop import _request as _runtime_request
from scheduler.test_pipeline_loop import _runtime
from server.fakes import FakeEngine, FakeEngineBackend, engine_config, request

from nari_qwen3_tts.engine.engine import Engine
from nari_qwen3_tts.engine.state import LiveInputState


def test_runtime_live_update_batch_is_one_atomic_version_transition(monkeypatch) -> None:
    runtime, _execution = _runtime()
    base = _runtime_request(1)
    continuation = replace(
        base.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))
    state = runtime.request(base.request_id)
    before_input = state.input
    before_version = state.generation.version

    continuation_type = type(continuation)
    original = continuation_type.append
    calls = 0

    def fail_second(self, token_ids, *, sequence, is_final):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second update rejected")
        return original(self, token_ids, sequence=sequence, is_final=is_final)

    monkeypatch.setattr(continuation_type, "append", fail_second)
    updates = (
        (torch.tensor([41, 42]), 0, False),
        (torch.tensor([43]), 1, True),
    )

    with pytest.raises(RuntimeError, match="second update"):
        runtime.update_request_input_batch(base.request_id, updates)

    assert state.input is before_input
    assert state.generation.version == before_version


def test_runtime_live_update_batch_commits_all_chunks_once() -> None:
    runtime, _execution = _runtime()
    base = _runtime_request(1)
    continuation = replace(
        base.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))
    state = runtime.request(base.request_id)
    before_version = state.generation.version

    runtime.update_request_input_batch(
        base.request_id,
        (
            (torch.tensor([41, 42]), 0, False),
            (torch.tensor([43]), 1, True),
        ),
    )

    updated = state.input.talker_input.continuation
    assert updated.materialized_token_ids().tolist() == [41, 42, 43, 90]
    assert updated.next_update_sequence == 2
    assert updated.input_finished
    assert state.generation.version == before_version + 1


def test_validated_live_update_does_not_read_cuda_token_values_on_the_host(monkeypatch) -> None:
    runtime, _execution = _runtime()
    base = _runtime_request(1)
    continuation = replace(
        base.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))
    monkeypatch.setattr(
        "nari_qwen3_tts.engine.pipeline.torch.any",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validated live tokens must not trigger a host readback")
        ),
    )

    runtime.update_validated_request_input_batch(
        base.request_id,
        ((torch.tensor([41]), 0, False),),
    )

    continuation = runtime.request(base.request_id).input.talker_input.continuation
    assert continuation.materialized_token_ids().tolist() == [41]


def test_engine_validates_live_token_ids_before_device_staging(monkeypatch) -> None:
    runtime, _execution = _runtime()
    base = _runtime_request(1)
    continuation = replace(
        base.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))
    engine = object.__new__(Engine)
    engine.pipeline = runtime
    monkeypatch.setattr(
        "nari_qwen3_tts.engine.engine.torch.tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid IDs reached device staging")
        ),
    )

    with pytest.raises(ValueError, match="invalid text token"):
        engine._update_request_input_batch(base.request_id, (((-1,), 0, False),))


def test_engine_keeps_live_append_tokens_on_the_host_until_decode_staging() -> None:
    runtime, _execution = _runtime()
    base = _runtime_request(1)
    continuation = replace(
        base.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long, device="meta"),
        pad_token_id=torch.tensor([0], dtype=torch.long, device="meta"),
        input_finished=False,
        terminal_token_id=torch.tensor([90], dtype=torch.long, device="meta"),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))
    engine = object.__new__(Engine)
    engine.pipeline = runtime

    engine._update_request_input_batch(base.request_id, (((41,), 0, False),))

    updated = runtime.request(base.request_id).input.talker_input.continuation
    assert updated.token_at(0).device.type == "cpu"


def test_engine_uses_one_transaction_for_each_pre_tokenized_live_update() -> None:
    class TransactionPort(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.transactions: list[tuple[str, tuple[tuple[tuple[int, ...], int, bool], ...]]] = []

        def update_request_input_batch(self, request_id, updates) -> None:
            normalized = tuple(
                (tuple(token_ids), sequence, is_final)
                for token_ids, sequence, is_final in updates
            )
            self.transactions.append((request_id, normalized))
            for token_ids, sequence, is_final in normalized:
                super().update_request_input(
                    request_id,
                    token_ids,
                    sequence=sequence,
                    is_final=is_final,
                )

    port = TransactionPort()
    engine = FakeEngine(
        port,
        config=engine_config(max_update_tokens=2, max_buffered_bytes=32),
    )
    engine.start(request("probe"), timeout_s=2)
    try:
        engine.begin_live(
            "live",
            request("ab"),
            initial_token_ids=(11,),
            initial_wrapped_ids=(101, 102, 103, 11, 104),
            input_finished=False,
        )
        record = engine._records["live"]
        assert engine.append_text("live", (12, 13), sequence=0, is_final=True) is None
        state = record.live_state
        assert isinstance(state, LiveInputState)
        assert state.next_engine_sequence == 1
        assert state.input_finished
        assert len(port.transactions) == 1
        assert port.transactions[0][1] == (((12, 13), 0, True),)
    finally:
        try:
            engine.cancel("live", timeout_s=0.5)
        except KeyError:
            pass
        engine.stop(timeout_s=2)


def test_engine_append_reply_confirms_queue_acceptance_before_safe_point_publication() -> None:
    class DeferredUpdateBackend(FakeEngineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.update_received = threading.Event()
            self.allow_publication = threading.Event()
            self.receipt: Future[None] | None = None

        def update_request_input_batch(self, request_id, updates):
            for token_ids, sequence, is_final in updates:
                super().update_request_input(
                    request_id,
                    token_ids,
                    sequence=sequence,
                    is_final=is_final,
                )
            self.receipt = Future()
            self.update_received.set()
            return self.receipt

        def step(self, *, now_s):
            receipt = self.receipt
            if (
                receipt is not None
                and not receipt.done()
                and self.allow_publication.is_set()
            ):
                receipt.set_result(None)
                return object()
            return super().step(now_s=now_s)

    backend = DeferredUpdateBackend()
    engine = FakeEngine(
        backend,
        config=engine_config(max_buffered_bytes=32, owner_poll_interval_s=0.0005),
    )
    engine.start(request("probe"), timeout_s=2)
    try:
        engine.begin_live(
            "live",
            request("ab"),
            initial_token_ids=(11,),
            initial_wrapped_ids=(101, 102, 103, 11, 104),
            input_finished=False,
        )
        engine.append_text(
            "live",
            (12,),
            sequence=0,
            is_final=False,
            timeout_s=1,
        )
        assert backend.update_received.wait(timeout=1)
        assert backend.receipt is not None
        assert not backend.receipt.done(), "safe-point publication unexpectedly completed"
    finally:
        backend.allow_publication.set()
        try:
            engine.cancel("live", timeout_s=0.5)
        except KeyError:
            pass
        engine.stop(timeout_s=2)

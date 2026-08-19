from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace

import pytest
import torch
from scheduler.test_pipeline_loop import _request, _runtime

from nari_qwen3_tts.contract import SynthesisStage
from nari_qwen3_tts.contract.errors import LiveInputClosedError, RequestCancelled
from nari_qwen3_tts.contract.request import TextContinuation
from nari_qwen3_tts.engine.pipeline import MAX_PENDING_LIVE_TEXT_TOKENS
from nari_qwen3_tts.engine.state import GenerationPhase


def _claim_generation(runtime):
    candidates = tuple(
        work
        for work in runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
        if work.stage.lane.value == "generation"
    )
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    prepared = runtime._prepare_execution(decision)
    return prepared, runtime._claim_decision(decision)


def test_unfinished_continuation_waits_without_blocking_codec_lane() -> None:
    runtime, _execution = _runtime()
    base = _request(1)
    continuation = replace(
        base.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))

    runtime.step(now_s=0.0)  # Talker prefill
    runtime.step(now_s=0.1)  # Code Predictor produces one Codec frame
    snapshot = runtime.planner.candidates(runtime.state_store.requests, now_s=0.2)
    assert {(item.request_id, item.stage) for item in snapshot} == {
        (base.request_id, SynthesisStage.CODEC)
    }


def test_ready_work_probe_does_not_advance_snapshot_observation_state() -> None:
    runtime, _execution = _runtime()
    runtime.admit(_request(1))

    before_sequence = runtime.planner._snapshot_sequence
    first = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    assert runtime.has_ready_work()
    second = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)

    assert runtime.planner._snapshot_sequence == before_sequence
    assert tuple(item.identity for item in second) == tuple(item.identity for item in first)
    assert tuple(item.ready_sequence for item in second) == tuple(
        item.ready_sequence for item in first
    )


def test_live_updates_are_ordered_and_final_adds_exactly_one_text_eos() -> None:
    runtime, _execution = _runtime()
    base = _request(1)
    continuation = TextContinuation(
        non_streaming_mode=False,
        token_ids=torch.empty(0, dtype=torch.long),
        pad_token_id=base.talker_input.continuation.pad_token_id,
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))

    runtime.update_request_input(base.request_id, torch.tensor([41, 42]), sequence=0, is_final=False)
    updated = runtime.request(base.request_id).input.talker_input.continuation
    assert updated.materialized_token_ids().tolist() == [41, 42]
    assert not updated.input_finished
    assert updated.next_update_sequence == 1

    with pytest.raises(ValueError, match="sequence"):
        runtime.update_request_input(base.request_id, torch.tensor([99]), sequence=0, is_final=False)

    runtime.update_request_input(base.request_id, torch.tensor([43]), sequence=1, is_final=True)
    final = runtime.request(base.request_id).input.talker_input.continuation
    assert final.materialized_token_ids().tolist() == [41, 42, 43, 90]
    assert final.input_finished
    with pytest.raises(RuntimeError, match="finished"):
        runtime.update_request_input(base.request_id, torch.tensor([44]), sequence=2, is_final=False)


def test_live_update_rejects_normal_requests_and_waits_for_inflight_generation() -> None:
    runtime, _execution = _runtime()
    normal = _request(0)
    runtime.admit(normal)
    with pytest.raises(RuntimeError, match="finished"):
        runtime.update_request_input(normal.request_id, torch.tensor([1]), sequence=0, is_final=False)

    live = _request(1)
    continuation = replace(
        live.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(live, talker_input=replace(live.talker_input, continuation=continuation)))
    prepared, claim = _claim_generation(runtime)
    state = runtime.request(live.request_id)
    before_input = state.input
    before_version = state.generation.version

    receipt = runtime.update_request_input(
        live.request_id,
        torch.tensor([1]),
        sequence=0,
        is_final=False,
    )
    published_versions: list[int] = []
    receipt.add_done_callback(lambda _receipt: published_versions.append(state.generation.version))

    assert not receipt.done()
    assert state.input is before_input
    assert state.generation.version == before_version

    runtime.execute_decision(prepared, claim)
    assert not receipt.done()
    runtime.step(now_s=0.1)

    receipt.result(timeout=0)
    published = state.input.talker_input.continuation
    assert published.materialized_token_ids().tolist() == [1]
    assert published.next_update_sequence == 1
    assert published_versions == [before_version + 2]


def test_multiple_inflight_live_updates_publish_atomically_in_sequence() -> None:
    runtime, _execution = _runtime()
    live = _request(1)
    continuation = replace(
        live.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(live, talker_input=replace(live.talker_input, continuation=continuation)))
    state = runtime.request(live.request_id)
    before_input = state.input
    before_version = state.generation.version
    prepared, claim = _claim_generation(runtime)

    first = runtime.update_request_input(
        live.request_id,
        torch.tensor([41, 42]),
        sequence=0,
        is_final=False,
    )
    final = runtime.update_request_input(
        live.request_id,
        torch.tensor([43]),
        sequence=1,
        is_final=True,
    )
    published_versions: list[int] = []
    final.add_done_callback(lambda _receipt: published_versions.append(state.generation.version))

    assert not first.done() and not final.done()
    assert state.input is before_input
    assert state.generation.version == before_version
    with pytest.raises((ValueError, RuntimeError), match="sequence|finished"):
        runtime.update_request_input(
            live.request_id,
            torch.tensor([44]),
            sequence=1,
            is_final=False,
        )

    runtime.execute_decision(prepared, claim)
    runtime.step(now_s=0.1)

    first.result(timeout=0)
    final.result(timeout=0)
    published = state.input.talker_input.continuation
    assert published.materialized_token_ids().tolist() == [41, 42, 43, 90]
    assert published.input_finished
    assert published.next_update_sequence == 2
    assert published_versions == [before_version + 2]


def test_cancelling_inflight_generation_fails_unpublished_live_input() -> None:
    runtime, _execution = _runtime()
    live = _request(1)
    continuation = replace(
        live.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(live, talker_input=replace(live.talker_input, continuation=continuation)))
    state = runtime.request(live.request_id)
    before_input = state.input
    prepared, claim = _claim_generation(runtime)

    receipt = runtime.update_request_input(
        live.request_id,
        torch.tensor([41]),
        sequence=0,
        is_final=False,
    )
    runtime.cancel(live.request_id)

    with pytest.raises(RequestCancelled, match="cancel"):
        receipt.result(timeout=0)
    assert state.input is before_input

    runtime.execute_decision(prepared, claim)
    assert state.pending_live_input is None


def test_generation_finishing_at_safe_point_rejects_unpublished_live_input() -> None:
    runtime, _execution = _runtime()
    live = _request(1)
    continuation = replace(
        live.talker_input.continuation,
        token_ids=torch.tensor([11, 12]),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(live, talker_input=replace(live.talker_input, continuation=continuation)))
    state = runtime.request(live.request_id)
    state.generation.phase = GenerationPhase.TALKER_DECODE
    state.generation.generation_step = 1
    state.generation.step_input = torch.zeros(4)
    before_continuation = state.input.talker_input.continuation
    before_version = state.generation.version
    prepared, claim = _claim_generation(runtime)

    receipt = runtime.update_request_input(
        live.request_id,
        torch.tensor([41]),
        sequence=0,
        is_final=False,
    )
    runtime.execute_decision(prepared, claim)
    runtime.step(now_s=0.1)

    with pytest.raises(LiveInputClosedError, match="finished before live input publication"):
        receipt.result(timeout=0)
    assert state.generation.phase is GenerationPhase.DONE
    assert state.input.talker_input.continuation is before_continuation
    assert state.generation.version == before_version + 1
    assert state.pending_live_input is None


def test_pending_live_input_capacity_includes_current_and_staged_tokens() -> None:
    runtime, _execution = _runtime()
    live = _request(1)
    capacity = MAX_PENDING_LIVE_TEXT_TOKENS
    continuation = replace(
        live.talker_input.continuation,
        token_ids=torch.ones(capacity - 1, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(live, talker_input=replace(live.talker_input, continuation=continuation)))
    _prepared, _claim = _claim_generation(runtime)

    receipt = runtime.update_request_input(
        live.request_id,
        torch.tensor([1]),
        sequence=0,
        is_final=False,
    )
    assert not receipt.done()
    with pytest.raises(ValueError, match="capacity"):
        runtime.update_request_input(
            live.request_id,
            torch.tensor([1]),
            sequence=1,
            is_final=False,
        )


def test_engine_append_command_reply_waits_for_pipeline_publication() -> None:
    runtime, _execution = _runtime()
    live = _request(1)
    continuation = replace(
        live.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(live, talker_input=replace(live.talker_input, continuation=continuation)))
    prepared, claim = _claim_generation(runtime)
    reply: Future[int] = Future()

    receipt = runtime.update_request_input_batch(
        live.request_id,
        ((torch.tensor([41]), 0, False),),
    )
    receipt.add_done_callback(
        lambda completed: reply.set_result(1)
        if completed.exception() is None
        else reply.set_exception(completed.exception())
    )

    assert not reply.done()
    runtime.execute_decision(prepared, claim)
    assert not reply.done()
    runtime.step(now_s=0.1)
    assert reply.result(timeout=0) == 1


def test_live_input_suppresses_talker_eos_until_text_eos_is_consumed() -> None:
    runtime, execution = _runtime()
    base = _request(1)
    unfinished = replace(
        base.talker_input.continuation,
        token_ids=torch.tensor([41, 42, 43]),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=unfinished)))
    state = runtime.request(base.request_id)
    state.generation.phase = GenerationPhase.TALKER_DECODE
    state.generation.generation_step = 2
    state.generation.step_input = torch.zeros(4)
    runtime.step(now_s=0.0)
    assert [row.suppress_eos for row in execution.calls[-1][2].rows] == [True]

    runtime, execution = _runtime()
    finished = replace(unfinished, token_ids=torch.tensor([41, 42, 90]), input_finished=True)
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=finished)))
    state = runtime.request(base.request_id)
    state.generation.phase = GenerationPhase.TALKER_DECODE
    state.generation.generation_step = 2
    state.generation.step_input = torch.zeros(4)
    runtime.step(now_s=0.0)
    assert [row.suppress_eos for row in execution.calls[-1][2].rows] == [False]


def test_live_update_rejects_empty_invalid_and_finished_generation() -> None:
    runtime, _execution = _runtime()
    base = _request(1)
    continuation = replace(
        base.talker_input.continuation,
        token_ids=torch.empty(0, dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    runtime.admit(replace(base, talker_input=replace(base.talker_input, continuation=continuation)))
    with pytest.raises(ValueError, match="empty"):
        runtime.update_request_input(base.request_id, torch.empty(0, dtype=torch.long), sequence=0, is_final=False)
    with pytest.raises(ValueError, match="invalid"):
        runtime.update_request_input(base.request_id, torch.tensor([200_000]), sequence=0, is_final=False)
    runtime.request(base.request_id).generation.phase = GenerationPhase.DONE
    with pytest.raises(RuntimeError, match="finished"):
        runtime.update_request_input(base.request_id, torch.tensor([1]), sequence=0, is_final=False)

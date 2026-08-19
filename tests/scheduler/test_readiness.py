from __future__ import annotations

from dataclasses import replace

import torch

from nari_qwen3_tts.contract import (
    CodecBatchCompatibility as CodecCompatibility,
)
from nari_qwen3_tts.contract import CodecExecutionMode, SynthesisStage
from nari_qwen3_tts.contract.request import TextContinuation
from nari_qwen3_tts.engine.state import CodecPhase, GenerationPhase
from scheduler.test_pipeline_loop import _request, _runtime


def _projection(work) -> tuple[object, ...]:
    return (
        work.request_id,
        getattr(work, "request_version", getattr(work, "version", None)),
        work.lane,
        work.stage,
        work.logical_step,
        work.compatibility,
        work.admission_sequence,
        work.startup,
        getattr(work, "playback_deadline_s", getattr(work, "deadline_s", None)),
        getattr(work, "execution_reserve_s", getattr(work, "reserve_s", None)),
    )


def _planner_projection(runtime) -> tuple[tuple[object, ...], ...]:
    return tuple(
        map(
            _projection,
            runtime.planner.candidates(runtime.state_store.requests, now_s=17.0),
        )
    )


def test_planner_readiness_covers_all_request_states_without_mutation() -> None:
    runtime, _execution = _runtime()
    for index in range(7):
        runtime.admit(_request(index))

    runtime.request("request-1").cancel_requested = True

    code_predictor = runtime.request("request-2")
    code_predictor.generation.phase = GenerationPhase.CODE_PREDICTOR

    decode = runtime.request("request-3")
    decode.generation.phase = GenerationPhase.TALKER_DECODE

    waiting = runtime.request("request-4")
    waiting.generation.phase = GenerationPhase.TALKER_DECODE
    waiting_input = waiting.input
    continuation = TextContinuation(
        non_streaming_mode=False,
        token_ids=torch.empty(0, dtype=torch.long),
        pad_token_id=torch.tensor([99]),
        input_finished=False,
        terminal_token_id=torch.tensor([90]),
    )
    waiting.input = replace(
        waiting_input,
        request=replace(waiting_input.request, non_streaming_mode=False),
        talker_input=replace(waiting_input.talker_input, continuation=continuation),
    )

    codec = runtime.request("request-5")
    codec.generation.phase = GenerationPhase.DONE
    codec.codec.phase = CodecPhase.READY
    codec.codec.ready_compatibility = CodecCompatibility(
        mode=CodecExecutionMode.WARM,
        model_frames=4,
        input_frames=4,
        visible_frames=4,
        pcm_start_frame=0,
        producer_frames=4,
        terminal=False,
    )
    codec.codec.execution_reserve_s = 0.025
    codec.codec.playback_started_at_s = 10.0
    codec.codec.emitted_duration_s = 2.0

    in_flight = runtime.request("request-6")
    in_flight.generation.claim_token = 123

    before = tuple(state.committed_view() for state in runtime.state_store.requests)
    owners = tuple(
        (
            state.generation.claim_token,
            state.generation.claim_batch_id,
            state.codec.claim_token,
            state.codec.claim_batch_id,
        )
        for state in runtime.state_store.requests
    )

    planned = _planner_projection(runtime)
    assert tuple((work[0], work[3]) for work in planned) == (
        ("request-0", SynthesisStage.TALKER_PREFILL),
        ("request-2", SynthesisStage.CODE_PREDICTOR),
        ("request-3", SynthesisStage.TALKER_DECODE),
        ("request-5", SynthesisStage.CODEC),
    )
    assert planned[-1][7:] == (False, 12.0, 0.025)
    assert tuple(state.committed_view() for state in runtime.state_store.requests) == before
    assert tuple(
        (
            state.generation.claim_token,
            state.generation.claim_batch_id,
            state.codec.claim_token,
            state.codec.claim_batch_id,
        )
        for state in runtime.state_store.requests
    ) == owners


def test_runtime_uses_planner_readiness_across_generation_and_codec_progress() -> None:
    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    runtime.admit(_request(1))

    for turn in range(12):
        candidates = runtime.planner.candidates(
            runtime.state_store.requests,
            now_s=runtime._current_now_s,
        )
        assert tuple(map(_projection, candidates)) == _planner_projection(runtime)
        if not runtime.has_ready_work():
            break
        runtime.step(now_s=float(turn))

    assert runtime.request("request-0").generation.phase is GenerationPhase.DONE
    assert runtime.request("request-1").generation.phase is GenerationPhase.DONE
    assert all(
        work[3] in {
            SynthesisStage.TALKER_PREFILL,
            SynthesisStage.TALKER_DECODE,
            SynthesisStage.CODE_PREDICTOR,
            SynthesisStage.CODEC,
        }
        for work in _planner_projection(runtime)
    )

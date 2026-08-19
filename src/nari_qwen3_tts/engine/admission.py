"""Request admission values assembled by the Engine composition path."""

from __future__ import annotations

from dataclasses import replace

from nari_qwen3_tts.contract.request import (
    AdmittedRequest,
    SynthesisRequest,
    TalkerPrompt,
)
from nari_qwen3_tts.executor.input_layout import TalkerInputPlan
from nari_qwen3_tts.model.capabilities import QWEN3_TTS_CAPABILITIES


def request_chunk_schedule(
    request: SynthesisRequest,
    execution_config,
    *,
    suppress_first_silent_frame: bool,
) -> tuple[int, ...]:
    if request.stream_chunk_schedule is not None:
        return request.stream_chunk_schedule
    adaptive = (
        request.stream_first_chunk_frames is not None
        or request.stream_steady_chunk_frames is not None
    )
    if request.stream_chunk_frames is not None and not adaptive:
        return (request.stream_chunk_frames,)
    if adaptive:
        first = request.stream_first_chunk_frames or 4
        steady = request.stream_steady_chunk_frames or request.stream_chunk_frames or 12
        return (first, steady)
    return execution_config.codec_chunks(
        silent_bootstrap_suppressed=suppress_first_silent_frame
    )


def make_admitted_request(
    *,
    request_id: str,
    request: SynthesisRequest,
    talker_plan: TalkerInputPlan,
    execution_config,
    admitted_at_s: float,
    input_finished: bool = True,
) -> AdmittedRequest:
    suppress = QWEN3_TTS_CAPABILITIES.can_suppress_fixed_bootstrap_audio(request)
    planned_continuation = talker_plan.continuations[0]
    continuation_tokens = planned_continuation.token_ids
    terminal_token = None
    if not input_finished:
        if request.non_streaming_mode:
            raise ValueError("live input requires non_streaming_mode=False")
        terminal_token = continuation_tokens[-1:].clone()
        continuation_tokens = continuation_tokens[:-1]
    return AdmittedRequest(
        request_id=request_id,
        request=request,
        talker_input=TalkerPrompt(
            text_token_ids=talker_plan.text_token_ids,
            codec_token_ids=talker_plan.codec_token_ids,
            codec_token_mask=talker_plan.codec_token_mask,
            sequence_length=talker_plan.sequence_lengths[0],
            continuation=replace(
                planned_continuation,
                token_ids=continuation_tokens,
                input_finished=input_finished,
                terminal_token_id=terminal_token,
            ),
        ),
        chunk_schedule=request_chunk_schedule(
            request,
            execution_config,
            suppress_first_silent_frame=suppress,
        ),
        suppress_first_silent_frame=suppress,
        admitted_at_s=admitted_at_s,
    )


__all__ = ["make_admitted_request", "request_chunk_schedule"]

"""Pure Codec readiness derivation for Engine-owned mutable state."""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.contract.frames import WHOLE_SEQUENCE_MAX_FRAMES
from nari_qwen3_tts.contract.stage import CodecExecutionMode
from nari_qwen3_tts.contract.work import CodecBatchCompatibility
from nari_qwen3_tts.engine.state import CodecLane, CodecPhase

WHOLE_SEQUENCE_EXECUTION_RESERVE_S = 0.010
INCREMENTAL_CODEC_EXECUTION_RESERVE_S = 0.025


@dataclass(frozen=True, slots=True)
class CodecReadiness:
    compatibility: CodecBatchCompatibility | None
    phase: CodecPhase
    execution_reserve_s: float


def next_codec_readiness(lane: CodecLane) -> CodecReadiness | None:
    """Compute the next Codec work without mutating request state."""
    if (
        lane.phase is CodecPhase.DONE
        or lane.ready_compatibility is not None
        or lane.claim_token is not None
    ):
        return None
    buffered = len(lane.buffered_frames)
    next_size = lane.chunk_schedule[min(lane.chunk_index, len(lane.chunk_schedule) - 1)]
    if lane.producer_done:
        if buffered == 0:
            return CodecReadiness(
                CodecBatchCompatibility(
                    mode=CodecExecutionMode.EMPTY,
                    model_frames=0,
                    input_frames=0,
                    visible_frames=0,
                    pcm_start_frame=0,
                    terminal=True,
                    producer_frames=0,
                ),
                CodecPhase.READY,
                INCREMENTAL_CODEC_EXECUTION_RESERVE_S,
            )
        producer_frames = min(buffered, next_size)
        terminal = producer_frames == buffered
    else:
        if buffered < next_size:
            return CodecReadiness(None, CodecPhase.COLLECTING, 0.0)
        producer_frames = next_size
        terminal = False

    if lane.decoder_state is not None:
        mode = CodecExecutionMode.WARM
        input_frames = producer_frames
        model_frames = producer_frames
        if terminal:
            model_frames = next(
                (size for size in lane.terminal_pad_frames if size >= input_frames),
                input_frames,
            )
        pcm_start_frame = 0
        visible_frames = producer_frames
    else:
        input_frames = lane.context_frames_consumed + producer_frames
        model_frames = input_frames
        mode = (
            CodecExecutionMode.WHOLE_SEQUENCE
            if input_frames <= WHOLE_SEQUENCE_MAX_FRAMES
            else CodecExecutionMode.COLD
        )
        if terminal and mode is CodecExecutionMode.COLD:
            model_frames = next(
                (size for size in lane.cold_terminal_pad_frames if size >= input_frames),
                input_frames,
            )
        pcm_start_frame = lane.context_frames_consumed
        visible_frames = producer_frames
        if lane.context_frames_consumed == 0 and lane.suppress_first_silent_frame:
            pcm_start_frame = 1
            visible_frames = max(0, input_frames - 1)
    compatibility = CodecBatchCompatibility(
        mode=mode,
        model_frames=model_frames,
        input_frames=input_frames,
        visible_frames=visible_frames,
        pcm_start_frame=pcm_start_frame,
        terminal=terminal,
        producer_frames=producer_frames,
    )
    reserve_s = (
        WHOLE_SEQUENCE_EXECUTION_RESERVE_S
        if mode is CodecExecutionMode.WHOLE_SEQUENCE
        else INCREMENTAL_CODEC_EXECUTION_RESERVE_S
    )
    return CodecReadiness(compatibility, CodecPhase.READY, reserve_s)


__all__ = ["CodecReadiness", "next_codec_readiness"]

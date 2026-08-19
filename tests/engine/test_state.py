from __future__ import annotations

import pytest
import torch


def test_request_state_normal_and_cancelled_removal_require_all_lifetimes() -> None:
    from nari_qwen3_tts.engine.state import (
        CodecPhase,
        GenerationPhase,
        RequestState,
    )
    state = RequestState("request-1", input=None)
    state.generation.phase = GenerationPhase.DONE
    state.codec.phase = CodecPhase.DONE
    state.codec.compute_terminal = True
    state.codec.output_terminal = True
    assert state.is_removable

    state.pending_gpu_submissions = 1
    assert not state.is_removable
    state.pending_gpu_submissions = 0
    state.codec.pending_outputs = 1
    assert not state.is_removable
    state.codec.pending_outputs = 0
    state.generation.claim_token = 1
    assert not state.is_removable

    cancelled = RequestState("request-2", input=None)
    cancelled.cancel_requested = True
    assert cancelled.is_removable


def test_codec_lane_playback_clock_is_monotonic_and_pcm16_exact() -> None:
    from nari_qwen3_tts.engine.state import CodecLane

    lane = CodecLane()
    lane.mark_routed(4_800, at_s=3.0, sample_rate=24_000)
    assert lane.playback_started_at_s == 3.0
    assert lane.last_routed_at_s == 3.0
    assert lane.emitted_duration_s == pytest.approx(0.1)

    lane.mark_routed(0, at_s=3.1, sample_rate=24_000)
    assert lane.playback_started_at_s == 3.0
    with pytest.raises(ValueError, match="monotonic"):
        lane.mark_routed(2, at_s=2.9, sample_rate=24_000)
    with pytest.raises(ValueError, match="PCM16"):
        lane.mark_routed(3, at_s=3.2, sample_rate=24_000)


def test_codec_lane_consumes_owned_frame_prefix_only() -> None:
    from nari_qwen3_tts.engine.state import CodecLane

    frames = tuple(torch.tensor([index]) for index in range(3))
    lane = CodecLane(buffered_frames=frames)
    lane.consume(2)
    assert lane.buffered_frames == (frames[2],)
    assert lane.context_frames_consumed == 2
    assert lane.chunk_index == 1
    with pytest.raises(ValueError, match="exceeds"):
        lane.consume(2)

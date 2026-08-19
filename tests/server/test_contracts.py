from __future__ import annotations

import pytest

from nari_qwen3_tts.contract.errors import BackpressureExceeded
from nari_qwen3_tts.contract.health import (
    EngineHealth,
    ServicePhase,
    StageStats,
    evaluate_readiness,
)
from nari_qwen3_tts.contract.stream import PCMStream


def _health(*, submitted: int = 0) -> EngineHealth:
    return EngineHealth(
        required_keys=4,
        captured_keys=4,
        required_cuda_graph_instances=7,
        captured_cuda_graph_instances=7,
        capture_failures=0,
        eager_fallbacks=0,
        stages=tuple(
            StageStats(name, submitted, submitted, 0)
            for name in ("talker_prefill", "talker_decode", "code_predictor", "codec")
        ),
    )


def test_capture_health_without_ordinary_tts_is_not_ready() -> None:
    verdict = evaluate_readiness(
        phase=ServicePhase.STARTING,
        health=_health(),
        ordinary_pcm_bytes=0,
        ordinary_stage_deltas={},
        ordinary_retired=False,
    )
    assert not verdict.ready
    assert verdict.reason == "ordinary_tts_not_proven"


def test_readiness_requires_all_four_captured_stages_and_retirement() -> None:
    stage_deltas = {
        "talker_prefill": 1,
        "talker_decode": 2,
        "code_predictor": 3,
        "codec": 1,
    }
    not_retired = evaluate_readiness(
        phase=ServicePhase.STARTING,
        health=_health(submitted=1),
        ordinary_pcm_bytes=4,
        ordinary_stage_deltas=stage_deltas,
        ordinary_retired=False,
    )
    assert not not_retired.ready
    assert not_retired.reason == "ordinary_tts_not_retired"

    ready = evaluate_readiness(
        phase=ServicePhase.READY,
        health=_health(submitted=1),
        ordinary_pcm_bytes=4,
        ordinary_stage_deltas=stage_deltas,
        ordinary_retired=True,
    )
    assert ready.ready
    assert ready.reason == "ready"


def test_pcm_stream_budget_and_terminal_are_fail_closed() -> None:
    stream = PCMStream(max_buffered_bytes=4)
    stream.publish(b"12")
    stream.publish(b"34")
    assert stream.buffered_bytes == 4
    with pytest.raises(BackpressureExceeded):
        stream.publish(b"56")
    assert stream.buffered_bytes == 0
    with pytest.raises(BackpressureExceeded):
        stream.read(timeout_s=0)
    with pytest.raises(RuntimeError, match="terminal"):
        stream.close()


def test_pcm_stream_preserves_chunk_order_and_one_terminal() -> None:
    stream = PCMStream(max_buffered_bytes=8)
    stream.publish(b"12")
    stream.publish(b"")
    stream.close()
    assert stream.read(timeout_s=0) == b"12"
    assert stream.read(timeout_s=0) == b""
    assert stream.read(timeout_s=0) is None
    with pytest.raises(RuntimeError, match="terminal"):
        stream.close()

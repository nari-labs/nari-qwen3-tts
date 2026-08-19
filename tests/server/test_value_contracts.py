from __future__ import annotations

import math

import pytest

from nari_qwen3_tts.config import EngineConfig
from nari_qwen3_tts.contract.health import (
    EngineHealth,
    ServicePhase,
    StageStats,
    evaluate_readiness,
)
from nari_qwen3_tts.contract.stream import PCMStream

_STAGES = ("talker_prefill", "talker_decode", "code_predictor", "codec")


def _health() -> EngineHealth:
    return EngineHealth(
        required_keys=4,
        captured_keys=4,
        required_cuda_graph_instances=4,
        captured_cuda_graph_instances=4,
        capture_failures=0,
        eager_fallbacks=0,
        stages=tuple(StageStats(name, 1, 1, 0) for name in _STAGES),
    )


@pytest.mark.parametrize("value", [True, -1, 1.0])
def test_stage_execution_stats_rejects_non_integer_or_negative_counts(value) -> None:
    """malformed accounting cannot make capture health look ready."""

    with pytest.raises((TypeError, ValueError), match="submitted|count|non-negative"):
        StageStats("talker_prefill", value, 0, 0)


@pytest.mark.parametrize("name", ["", "talker", "codec_warm", 1])
def test_stage_execution_stats_rejects_unknown_stage_names(name) -> None:
    """serving health has exactly the four fixed stage names."""

    with pytest.raises((TypeError, ValueError), match="stage|name"):
        StageStats(name, 0, 0, 0)


def test_stage_execution_stats_rejects_impossible_accounting() -> None:
    """replayed/failed work cannot exceed submitted work."""

    with pytest.raises(ValueError, match="account|submitted"):
        StageStats("codec", submitted=1, replayed=2, failed=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("required_keys", True),
        ("captured_keys", -1),
        ("required_cuda_graph_instances", 1.0),
        ("captured_cuda_graph_instances", -1),
        ("capture_failures", True),
        ("eager_fallbacks", -1),
    ],
)
def test_engine_health_rejects_malformed_counts(field: str, value) -> None:
    """CUDA Graph health values are typed non-negative integer statistics."""

    values = {
        "required_keys": 4,
        "captured_keys": 4,
        "required_cuda_graph_instances": 4,
        "captured_cuda_graph_instances": 4,
        "capture_failures": 0,
        "eager_fallbacks": 0,
        "stages": tuple(StageStats(name, 0, 0, 0) for name in _STAGES),
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match="health|count|non-negative"):
        EngineHealth(**values)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, 0.0, -0.1])
def test_owner_poll_interval_is_finite_positive_real(value) -> None:
    """invalid poll values cannot busy-spin or block forever."""

    with pytest.raises((TypeError, ValueError), match="poll|finite|positive"):
        EngineConfig(idle_poll_interval_s=value)


@pytest.mark.parametrize("bad_value", [True, 1.0, -1])
def test_readiness_rejects_non_integer_ordinary_stage_deltas(bad_value) -> None:
    """proxy/bool stage deltas are not ordinary model work."""

    deltas = dict.fromkeys(_STAGES, 1)
    deltas["codec"] = bad_value
    verdict = evaluate_readiness(
        phase=ServicePhase.READY,
        health=_health(),
        ordinary_pcm_bytes=4,
        ordinary_stage_deltas=deltas,
        ordinary_retired=True,
    )

    assert not verdict.ready
    assert verdict.reason == "ordinary_stage_coverage_incomplete"


@pytest.mark.parametrize("pcm_bytes", [True, 1.0, -2, 1, 3])
def test_readiness_rejects_malformed_or_unaligned_pcm_proof(pcm_bytes) -> None:
    """an ordinary proof is positive sample-aligned PCM16."""

    verdict = evaluate_readiness(
        phase=ServicePhase.READY,
        health=_health(),
        ordinary_pcm_bytes=pcm_bytes,
        ordinary_stage_deltas=dict.fromkeys(_STAGES, 1),
        ordinary_retired=True,
    )

    assert not verdict.ready
    assert verdict.reason == "ordinary_tts_not_proven"


def test_pcm_stream_rejects_odd_length_audio() -> None:
    """transport PCM is complete signed-16 samples."""

    stream = PCMStream(max_buffered_bytes=8)

    with pytest.raises(ValueError, match="PCM16|sample|even"):
        stream.publish(b"\x00")


def test_stream_failure_cannot_be_replaced_by_a_later_terminal() -> None:
    """first terminal cause remains authoritative."""

    from nari_qwen3_tts.contract.errors import RequestCancelled

    stream = PCMStream(max_buffered_bytes=8)
    stream.fail(RequestCancelled("first"))
    stream.fail(RequestCancelled("second"))

    with pytest.raises(RequestCancelled, match="first"):
        stream.read(timeout_s=0)
    with pytest.raises(RuntimeError, match="terminal"):
        stream.close()

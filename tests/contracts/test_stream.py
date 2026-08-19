from __future__ import annotations

import threading

import pytest


def test_pcm_stream_and_errors_have_one_canonical_class_identity() -> None:
    from nari_qwen3_tts.contract.errors import (
        BackpressureExceeded,
        RequestCancelled,
        RequestRejected,
        ServiceCapacityExceeded,
        ServiceUnavailable,
        SynthesisError,
    )
    from nari_qwen3_tts.contract.stream import PCMStream

    assert len(
        {
            PCMStream,
            SynthesisError,
            ServiceUnavailable,
            ServiceCapacityExceeded,
            BackpressureExceeded,
            RequestCancelled,
            RequestRejected,
        }
    ) == 7


def test_pcm_stream_preserves_fifo_acknowledgement_and_budget() -> None:
    from nari_qwen3_tts.contract.errors import BackpressureExceeded
    from nari_qwen3_tts.contract.stream import PCMStream

    stream = PCMStream(max_buffered_bytes=6)
    stream.publish(b"\x01\x00")
    stream.publish(b"\x02\x00")
    first = stream.acquire(timeout_s=0)
    assert first == b"\x01\x00"
    assert stream.buffered_bytes == 4
    stream.acknowledge(first)
    assert stream.read(timeout_s=0) == b"\x02\x00"
    assert stream.buffered_bytes == 0

    stream.publish(b"\x03\x00\x04\x00")
    with pytest.raises(BackpressureExceeded):
        stream.publish(b"\x05\x00\x06\x00")
    assert stream.terminal


def test_pcm_stream_cross_thread_notify_and_detach_are_explicit() -> None:
    from nari_qwen3_tts.contract.stream import PCMStream

    stream = PCMStream(max_buffered_bytes=8)
    received: list[bytes | None] = []
    consumer = threading.Thread(target=lambda: received.append(stream.read(timeout_s=1.0)))
    consumer.start()
    stream.publish(b"\x01\x00")
    consumer.join(timeout=1.0)

    assert received == [b"\x01\x00"]
    assert not stream.detached
    stream.detach()
    assert stream.detached
    stream.detach()
    assert stream.detached

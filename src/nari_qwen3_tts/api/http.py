"""HTTP streaming and error translation helpers."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException

from nari_qwen3_tts.api.wav import wav_header
from nari_qwen3_tts.contract.errors import (
    BackpressureExceeded,
    RequestCancelled,
    RequestRejected,
    ServiceCapacityExceeded,
    ServiceUnavailable,
    SynthesisError,
)
from nari_qwen3_tts.contract.stream import PCMStream
from nari_qwen3_tts.engine.engine import Engine


def stream_bytes(
    engine: Engine,
    request_id: str,
    stream: PCMStream,
    *,
    response_format: str,
    command_timeout_s: float = 10.0,
) -> Iterator[bytes]:
    try:
        if response_format == "wav":
            yield wav_header(pcm_bytes=None)
        yield from stream.iter_bytes(timeout_s=None)
    finally:
        if not stream.terminal:
            try:
                engine.cancel(request_id, timeout_s=command_timeout_s)
            except SynthesisError:
                pass


def error_response(error: BaseException) -> HTTPException:
    if isinstance(error, ServiceUnavailable):
        return HTTPException(503, str(error))
    if isinstance(error, ServiceCapacityExceeded):
        return HTTPException(429, str(error))
    if isinstance(error, RequestRejected):
        return HTTPException(422, str(error))
    if isinstance(error, RequestCancelled):
        return HTTPException(499, str(error))
    if isinstance(error, BackpressureExceeded):
        return HTTPException(503, str(error))
    return HTTPException(500, "Qwen3-TTS serving request failed")


__all__ = ["error_response", "stream_bytes"]

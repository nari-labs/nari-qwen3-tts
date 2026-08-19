"""PCM/WAV wire constants and header construction."""

from __future__ import annotations

import struct

SAMPLE_RATE = 24_000


def wav_header(*, pcm_bytes: int | None) -> bytes:
    if pcm_bytes is not None:
        if isinstance(pcm_bytes, bool) or not isinstance(pcm_bytes, int):
            raise TypeError("PCM byte length must be an integer")
        if pcm_bytes < 0 or pcm_bytes % 2:
            raise ValueError("PCM byte length must be non-negative and even")
        if pcm_bytes > 0xFFFFFFFF - 36:
            raise ValueError("PCM byte length is not representable in a RIFF container")
    data_size = 0xFFFFFFFF if pcm_bytes is None else pcm_bytes
    riff_size = 0xFFFFFFFF if pcm_bytes is None else 36 + pcm_bytes
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        SAMPLE_RATE,
        SAMPLE_RATE * 2,
        2,
        16,
        b"data",
        data_size,
    )


__all__ = ["SAMPLE_RATE", "wav_header"]

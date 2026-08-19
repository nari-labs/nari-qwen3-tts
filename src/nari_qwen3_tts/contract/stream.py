"""Thread-safe bounded PCM delivery stream."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Iterator

from nari_qwen3_tts.contract.errors import BackpressureExceeded, SynthesisError


class PCMStream:
    """One request's ordered byte queue with an independent hard budget."""

    def __init__(self, *, max_buffered_bytes: int) -> None:
        if isinstance(max_buffered_bytes, bool) or not isinstance(max_buffered_bytes, int):
            raise TypeError("max_buffered_bytes must be an integer")
        if max_buffered_bytes < 1:
            raise ValueError("max_buffered_bytes must be positive")
        self.max_buffered_bytes = max_buffered_bytes
        self._items: deque[bytes] = deque()
        self._buffered_bytes = 0
        self._inflight = False
        self._inflight_value = b""
        self._terminal = False
        self._detached = False
        self._error: SynthesisError | None = None
        self._condition = threading.Condition()

    @property
    def buffered_bytes(self) -> int:
        with self._condition:
            return self._buffered_bytes

    @property
    def terminal(self) -> bool:
        with self._condition:
            return self._terminal

    @property
    def detached(self) -> bool:
        with self._condition:
            return self._detached

    def detach(self) -> None:
        with self._condition:
            self._detached = True
            self._condition.notify_all()

    def publish(self, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise TypeError("PCM stream items must be bytes")
        if len(value) % 2:
            raise ValueError("PCM16 stream items must contain complete even-byte samples")
        with self._condition:
            if self._terminal:
                raise RuntimeError("PCM stream is terminal")
            if self._buffered_bytes + len(value) > self.max_buffered_bytes:
                error = BackpressureExceeded("request PCM backpressure budget exceeded")
                queued_bytes = sum(len(item) for item in self._items)
                self._items.clear()
                self._buffered_bytes -= queued_bytes
                self._error = error
                self._terminal = True
                self._condition.notify_all()
                raise error
            self._items.append(value)
            self._buffered_bytes += len(value)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            if self._terminal:
                raise RuntimeError("PCM stream terminal was already published")
            self._terminal = True
            self._condition.notify_all()

    def fail(self, error: SynthesisError) -> None:
        if not isinstance(error, SynthesisError):
            raise TypeError("PCM stream failures must be typed serving errors")
        with self._condition:
            if self._terminal:
                return
            queued_bytes = sum(len(item) for item in self._items)
            self._items.clear()
            self._buffered_bytes -= queued_bytes
            self._error = error
            self._terminal = True
            self._condition.notify_all()

    def acquire(self, *, timeout_s: float | None = None) -> bytes | None:
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("stream timeout must be a finite non-negative number")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while self._inflight or (not self._items and not self._terminal):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for PCM stream data")
                self._condition.wait(remaining)
            if self._error is not None:
                raise self._error
            if self._items:
                value = self._items.popleft()
                self._inflight = True
                self._inflight_value = value
                return value
            return None

    def acknowledge(self, value: bytes) -> None:
        with self._condition:
            if not self._inflight or value != self._inflight_value:
                raise RuntimeError("PCM acknowledgement does not own the in-flight item")
            self._buffered_bytes -= len(self._inflight_value)
            if self._buffered_bytes < 0:
                raise RuntimeError("PCM outstanding byte count underflowed")
            self._inflight = False
            self._inflight_value = b""
            self._condition.notify_all()

    def read(self, *, timeout_s: float | None = None) -> bytes | None:
        value = self.acquire(timeout_s=timeout_s)
        if value is not None:
            self.acknowledge(value)
        return value

    def iter_bytes(self, *, timeout_s: float | None = None) -> Iterator[bytes]:
        while True:
            value = self.acquire(timeout_s=timeout_s)
            if value is None:
                return
            try:
                yield value
            finally:
                self.acknowledge(value)


__all__ = ["PCMStream"]

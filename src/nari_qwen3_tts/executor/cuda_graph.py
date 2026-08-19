"""Private CUDA Graph capture and typed static-buffer lease primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import torch

CapturedOutput = torch.Tensor | tuple[torch.Tensor, ...]


class CapturedCall(Protocol):
    def replay(self) -> CapturedOutput: ...


class CaptureDriver(Protocol):
    def capture(
        self,
        operation: Callable[[], CapturedOutput],
        *,
        after_warmup: Callable[[], None] | None = None,
    ) -> CapturedCall: ...


@dataclass(slots=True)
class _TorchCapturedCall:
    graph: torch.cuda.CUDAGraph
    output: CapturedOutput

    def replay(self) -> CapturedOutput:
        self.graph.replay()
        return self.output


class TorchCaptureDriver:
    def __init__(
        self,
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        memory_pool: tuple[int, int] | None = None,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Qwen3-TTS captured execution requires CUDA")
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.memory_pool = memory_pool or torch.cuda.graphs.graph_pool_handle()

    def _autocast(self):
        return torch.amp.autocast(
            "cuda",
            enabled=self.autocast_dtype is not None,
            dtype=self.autocast_dtype,
        )

    def capture(
        self,
        operation: Callable[[], CapturedOutput],
        *,
        after_warmup: Callable[[], None] | None = None,
    ) -> CapturedCall:
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        for _ in range(2):
            with self._autocast():
                operation()
            if after_warmup is not None:
                after_warmup()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with self._autocast(), torch.cuda.graph(graph, pool=self.memory_pool):
            output = operation()
        torch.cuda.synchronize(self.device)
        return _TorchCapturedCall(graph, output)


def stage_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    logical_rows: int,
    *,
    padding_value: int | float | bool = 0,
) -> None:
    if isinstance(logical_rows, bool) or not isinstance(logical_rows, int):
        raise TypeError("logical row count must be an integer")
    if logical_rows < 0 or logical_rows > destination.shape[0]:
        raise ValueError("logical row count exceeds the static destination")
    if source.shape[0] != logical_rows or source.shape[1:] != destination.shape[1:]:
        raise ValueError("source tensor shape does not match the logical static-buffer slice")
    if source.dtype != destination.dtype:
        raise ValueError("source tensor dtype does not match the static buffer dtype")
    if source.device != destination.device:
        raise ValueError("source tensor device does not match the static buffer device")
    destination.fill_(padding_value)
    if logical_rows:
        destination[:logical_rows].copy_(source)


class SlotBusyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SlotLease:
    generation: int


class SlotLeaseState:
    """Generation-checked exclusive lease for one fixed CUDA Graph buffer set.

    Executors own one of these per captured slot. Choosing *which* slot to use
    is a separate concern -- the Talker rotates over its capture slots itself --
    so this holds no capacity and never searches.
    """

    def __init__(self) -> None:
        self._busy = False
        self._generation = 0

    @property
    def available(self) -> bool:
        return not self._busy

    def reserve(self) -> SlotLease:
        if self._busy:
            raise SlotBusyError("static CUDA Graph slot is in flight")
        self._busy = True
        self._generation += 1
        return SlotLease(self._generation)

    def release(self, lease: SlotLease) -> None:
        if not self._busy or self._generation != lease.generation:
            raise SlotBusyError("stale or unowned static CUDA Graph slot lease")
        self._busy = False


class CudaGraphPoolFence:
    """Serialize submissions that share one CUDA Graph-private memory pool.

    CUDA Graph pool sharing permits allocations to alias across captures only
    when those CUDA Graphs are replayed in a single order. Two things enforce that:
    the lock, which makes a concurrent submission fail closed rather than wait,
    and the recorded event, which extends the lease interval through the
    asynchronous result copies before the next submission may touch aliased
    storage. The lease is only an identity token for the active holder.
    """

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self._lock = Lock()
        self._completion: torch.cuda.Event | None = None
        self._generation = 0
        self._active: SlotLease | None = None

    @property
    def available(self) -> bool:
        return self._active is None

    def reserve(self) -> SlotLease:
        if not self._lock.acquire(blocking=False):
            raise SlotBusyError("shared CUDA Graph memory pool is in flight")
        try:
            if self.device.type == "cuda" and self._completion is not None:
                torch.cuda.current_stream(self.device).wait_event(self._completion)
            self._generation += 1
            self._active = SlotLease(self._generation)
            return self._active
        except Exception:
            self._lock.release()
            raise

    def release(self, lease: SlotLease) -> None:
        if self._active is None or lease != self._active:
            raise SlotBusyError("stale or unowned static CUDA Graph slot lease")
        try:
            if self.device.type == "cuda":
                completion = torch.cuda.Event(blocking=False)
                completion.record(torch.cuda.current_stream(self.device))
                self._completion = completion
            self._active = None
        finally:
            self._lock.release()


class CudaSubmissionFence:
    """Completion handle for one already-enqueued CUDA submission boundary."""

    def __init__(self, event: object | None) -> None:
        if event is not None and not all(
            callable(getattr(event, name, None)) for name in ("query", "synchronize")
        ):
            raise TypeError("CUDA submission event must support query and synchronize")
        self._event = event

    @classmethod
    def completed(cls) -> CudaSubmissionFence:
        return cls(None)

    @classmethod
    def record(
        cls,
        *,
        device: torch.device,
        event_factory=None,
    ) -> CudaSubmissionFence:
        if device.type != "cuda":
            return cls.completed()
        factory = event_factory or (lambda: torch.cuda.Event(blocking=False))
        event = factory()
        event.record(torch.cuda.current_stream(device))
        return cls(event)

    def ready(self) -> bool:
        return self._event is None or bool(self._event.query())

    def wait(self) -> None:
        if self._event is not None:
            self._event.synchronize()


__all__ = [
    "CaptureDriver",
    "CapturedCall",
    "CapturedOutput",
    "CudaGraphPoolFence",
    "CudaSubmissionFence",
    "SlotBusyError",
    "SlotLease",
    "SlotLeaseState",
    "TorchCaptureDriver",
    "stage_rows",
]

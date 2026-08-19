"""Pinned-buffer PCM transfers started by the post-decision output poll."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PcmTransfer:
    """One lazy device-to-host copy and its pinned-buffer lease."""

    source: torch.Tensor
    host: torch.Tensor | None
    samples: int
    event: object
    pool: PcmTransferPool
    payload: bytes | None = None

    def poll(self) -> bytes | None:
        if self.payload is not None:
            return self.payload
        if not self.event.query():
            return None
        host = self.host
        if host is None:
            raise RuntimeError("ready PCM transfer lost its host storage")
        self.payload = host[: self.samples].numpy().tobytes()
        self.pool.release(host)
        self.host = None
        return self.payload

    def discard_when_ready(self) -> None:
        if self.payload is not None:
            return
        self.event.synchronize()
        if self.host is not None:
            self.pool.release(self.host)
            self.host = None

    def discard_if_ready(self) -> bool:
        """Release a cancelled transfer without synchronizing the Engine thread."""

        if self.payload is not None:
            return True
        if not self.event.query():
            return False
        if self.host is not None:
            self.pool.release(self.host)
            self.host = None
        return True


class PcmTransferPool:
    """Own dedicated D2H streams, CUDA events, and reusable pinned storage."""

    def __init__(
        self,
        *,
        maximum_samples: int,
        device: torch.device | None,
    ) -> None:
        if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int):
            raise TypeError("PCM host capacity must be an integer")
        if maximum_samples < 0:
            raise ValueError("PCM host capacity must be non-negative")
        self.maximum_samples = maximum_samples
        self.device = device
        self._streams: dict[int, torch.cuda.Stream] = {}
        self._hosts: list[torch.Tensor] = []
        self._active = 0

    @property
    def available(self) -> int:
        return len(self._hosts)

    @property
    def active(self) -> int:
        return self._active

    def prepare(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("PCM transfer capacity must be an integer")
        if capacity < 1:
            raise ValueError("PCM transfer capacity must be positive")
        if self.maximum_samples < 1:
            return
        if self._active:
            raise RuntimeError("PCM transfer pool cannot grow while copies are active")
        missing = capacity - len(self._hosts)
        self._hosts.extend(
            torch.empty(
                self.maximum_samples,
                dtype=torch.int16,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(max(0, missing))
        )
        if self.device is not None:
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            self._streams.setdefault(
                device_index,
                torch.cuda.Stream(device=self.device),
            )

    def _acquire(self, samples: int) -> torch.Tensor:
        if samples > self.maximum_samples:
            raise RuntimeError("Codec PCM exceeds the preallocated transfer capacity")
        if not self._hosts:
            raise RuntimeError("PCM transfer pool is exhausted")
        self._active += 1
        return self._hosts.pop()

    def release(self, host: torch.Tensor) -> None:
        if self._active < 1:
            raise RuntimeError("PCM transfer pool release underflow")
        self._hosts.append(host)
        self._active -= 1

    def begin(self, source: torch.Tensor) -> PcmTransfer:
        if not isinstance(source, torch.Tensor) or not source.is_cuda:
            raise TypeError("PCM transfer source must be a CUDA tensor")
        if source.dtype is not torch.int16:
            raise TypeError("Codec PCM must use torch.int16")
        flat = source.detach().contiguous().reshape(-1)
        host = self._acquire(flat.numel())
        device_index = source.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream = self._streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=source.device)
            self._streams[device_index] = stream
        stream.wait_stream(torch.cuda.current_stream(source.device))
        with torch.cuda.stream(stream):
            host[: flat.numel()].copy_(flat, non_blocking=True)
            event = torch.cuda.Event(blocking=False)
            event.record(stream)
        return PcmTransfer(flat, host, flat.numel(), event, self)


__all__ = ["PcmTransfer", "PcmTransferPool"]

"""Engine-owned request-order PCM queue with lazy D2H transfers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from nari_qwen3_tts.executor.pcm import PcmTransfer, PcmTransferPool


class _ReadyEvent:
    @staticmethod
    def synchronize() -> None:
        return None


@dataclass(slots=True)
class _HostTransfer:
    payload: bytes
    event: _ReadyEvent = _ReadyEvent()

    def poll(self) -> bytes:
        return self.payload

    def discard_when_ready(self) -> None:
        return None

    def discard_if_ready(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class PcmDelivery:
    request_id: str
    source: torch.Tensor
    value: bytes
    terminal_after: bool
    discarded: bool = False


@dataclass(slots=True)
class PendingPcm:
    request_id: str
    source: torch.Tensor
    transfer: PcmTransfer | _HostTransfer | None = None
    terminal_after: bool = False
    discarded: bool = False


class OutputQueue:
    """Retain committed PCM by request until a post-decision poll exposes it."""

    def __init__(self, pool: PcmTransferPool) -> None:
        self.pool = pool
        self._requests: dict[str, deque[PendingPcm]] = {}
        self.begin_count = 0

    @staticmethod
    def _validate(
        request_id: str,
        source: torch.Tensor,
        terminal_after: bool,
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("pending PCM requires a request ID")
        if not isinstance(source, torch.Tensor):
            raise TypeError("pending PCM source must be a tensor")
        if source.dtype is not torch.int16:
            raise TypeError("pending PCM source must use torch.int16")
        if type(terminal_after) is not bool:
            raise TypeError("pending PCM terminal marker must be a boolean")

    def enqueue(
        self,
        request_id: str,
        source: torch.Tensor,
        *,
        terminal_after: bool,
    ) -> None:
        self.enqueue_many(((request_id, source, terminal_after),))

    def enqueue_many(
        self,
        values: tuple[tuple[str, torch.Tensor, bool], ...],
    ) -> None:
        for request_id, source, terminal_after in values:
            self._validate(request_id, source, terminal_after)
        for request_id, source, terminal_after in values:
            self._requests.setdefault(request_id, deque()).append(
                PendingPcm(
                    request_id=request_id,
                    source=source,
                    terminal_after=terminal_after,
                )
            )

    def pending_metadata(
        self,
        request_order: tuple[str, ...],
    ) -> tuple[tuple[str, torch.Tensor, bool], ...]:
        return tuple(
            (request_id, pending.source, pending.terminal_after)
            for request_id in request_order
            for pending in self._requests.get(request_id, ())
        )

    def has_pending(self, request_id: str) -> bool:
        return bool(self._requests.get(request_id))

    def _pop_head(self, request_id: str) -> PendingPcm:
        pending = self._requests[request_id].popleft()
        if not self._requests[request_id]:
            del self._requests[request_id]
        return pending

    def _begin_transfer(self, pending: PendingPcm) -> None:
        if pending.transfer is not None or pending.discarded:
            return
        if pending.source.is_cuda or not isinstance(self.pool, PcmTransferPool):
            pending.transfer = self.pool.begin(pending.source)
        else:
            value = pending.source.detach().to(
                device="cpu",
                dtype=torch.int16,
            ).contiguous().reshape(-1)
            pending.transfer = _HostTransfer(value.numpy().tobytes())
        self.begin_count += 1

    def poll_ready(
        self,
        *,
        request_order: tuple[str, ...],
        block_oldest: bool = False,
    ) -> tuple[PcmDelivery, ...]:
        deliveries: list[PcmDelivery] = []
        oldest_pending: PendingPcm | None = None
        for request_id in request_order:
            request_values = self._requests.get(request_id)
            if not request_values:
                continue
            # Start every queued copy while the compute stream position is
            # still current. Publication remains a strict per-request FIFO:
            # only the contiguous ready prefix below may escape this poll.
            for pending in request_values:
                self._begin_transfer(pending)

            while request_values:
                pending = request_values[0]
                if pending.discarded:
                    if pending.transfer is not None and not pending.transfer.discard_if_ready():
                        if oldest_pending is None:
                            oldest_pending = pending
                        break
                    self._pop_head(request_id)
                    deliveries.append(
                        PcmDelivery(
                            request_id=request_id,
                            source=pending.source,
                            value=b"",
                            terminal_after=pending.terminal_after,
                            discarded=True,
                        )
                    )
                    continue
                assert pending.transfer is not None
                payload = pending.transfer.poll()
                if payload is None:
                    if oldest_pending is None:
                        oldest_pending = pending
                    break
                self._pop_head(request_id)
                deliveries.append(
                    PcmDelivery(
                        request_id=request_id,
                        source=pending.source,
                        value=payload,
                        terminal_after=pending.terminal_after,
                    )
                )
        if block_oldest and not deliveries and oldest_pending is not None:
            assert oldest_pending.transfer is not None
            oldest_pending.transfer.event.synchronize()
            return self.poll_ready(request_order=request_order)
        return tuple(deliveries)

    def cancel_request(self, request_id: str) -> None:
        """Suppress publication while retaining begun transfers until event-ready."""

        for pending in self._requests.get(request_id, ()):
            pending.discarded = True

    def discard_request(self, request_id: str) -> int:
        pending = self._requests.pop(request_id, ())
        for item in pending:
            if item.transfer is not None:
                item.transfer.discard_when_ready()
        return len(pending)


__all__ = ["OutputQueue", "PcmDelivery", "PendingPcm"]

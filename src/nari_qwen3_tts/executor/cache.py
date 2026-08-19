"""Request-owned page allocation and fail-closed pending Talker KV publication."""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.contract.result import KVPublication as _KVPublicationProposal
from nari_qwen3_tts.executor.cuda_graph import SlotBusyError


class KVAllocationError(RuntimeError):
    pass


class StaleKVPublicationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RequestKVState:
    request_id: str
    page_indices: tuple[int, ...] = ()
    sequence_length: int = 0
    position_start: int = 0
    version: int = 0


@dataclass(slots=True)
class _Pending:
    token: int
    prior: RequestKVState
    proposed_pages: tuple[int, ...]
    new_pages: tuple[int, ...]
    sequence_advance: int
    status: str = "pending"


@dataclass(frozen=True, slots=True)
class KVReservation:
    publications: tuple[PendingKVPublication, ...]


@dataclass(slots=True)
class PendingKVPublication:
    _allocator: PagedKVAllocator
    request_id: str
    token: int
    prior_version: int
    page_indices: tuple[int, ...]
    sequence_advance: int

    def proposal(self) -> _KVPublicationProposal:
        """Return the immutable Engine-facing state proposal for this reservation."""

        current = self._allocator.state(self.request_id)
        return _KVPublicationProposal(
            request_id=self.request_id,
            pages=self.page_indices,
            length=current.sequence_length + self.sequence_advance,
        )

    def validate(self) -> None:
        pending = self._allocator._pending.get(self.request_id)
        current = self._allocator._states.get(self.request_id)
        if (
            pending is None
            or current is None
            or pending.status != "pending"
            or pending.token != self.token
            or pending.prior.version != self.prior_version
            or current != pending.prior
            or pending.proposed_pages != self.page_indices
            or self.sequence_advance < 1
        ):
            raise StaleKVPublicationError("Talker KV publication is stale or no longer pending")

    def publish(self) -> None:
        self.validate()
        self._allocator._complete(self, publish=True)

    def discard(self) -> None:
        self.validate()
        self._allocator._complete(self, publish=False)


class PagedKVAllocator:
    """Small fixed-capacity allocator with no scheduling or transfer behavior."""

    def __init__(self, *, total_pages: int, page_size: int) -> None:
        if total_pages < 1 or page_size < 1:
            raise ValueError("paged KV capacity must be positive")
        self.total_pages = total_pages
        self.page_size = page_size
        self._free = list(range(total_pages))
        self._states: dict[str, RequestKVState] = {}
        self._pending: dict[str, _Pending] = {}
        self._next_token = 1

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def claim_scratch_pages(self, count: int) -> tuple[int, ...]:
        """Permanently remove executor-private padding pages from request capacity."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("scratch page count must be a positive integer")
        if count > len(self._free):
            raise KVAllocationError(f"cannot reserve {count} Talker scratch pages")
        pages = tuple(self._free[:count])
        del self._free[:count]
        return pages

    def add_request(self, request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if request_id in self._states:
            raise ValueError(f"request {request_id!r} already owns KV state")
        self._states[request_id] = RequestKVState(request_id)

    def state(self, request_id: str) -> RequestKVState:
        try:
            return self._states[request_id]
        except KeyError as error:
            raise KeyError(f"unknown Talker KV request {request_id!r}") from error

    def remove_request(self, request_id: str) -> None:
        if request_id in self._pending:
            raise SlotBusyError(f"request {request_id!r} KV is in flight")
        state = self.state(request_id)
        self._free.extend(state.page_indices)
        self._free.sort()
        del self._states[request_id]

    def reserve(self, request_ids: tuple[str, ...], advances: tuple[int, ...]) -> KVReservation:
        if not request_ids or len(request_ids) != len(advances):
            raise ValueError("KV reservation rows must be non-empty and aligned")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("KV reservation request rows must be unique")
        states = [self.state(request_id) for request_id in request_ids]
        for request_id, advance in zip(request_ids, advances, strict=True):
            if request_id in self._pending:
                raise SlotBusyError(f"request {request_id!r} KV is already in flight")
            if isinstance(advance, bool) or not isinstance(advance, int) or advance < 1:
                raise ValueError("KV sequence advances must be positive integers")

        allocated: list[int] = []
        created: list[str] = []
        publications: list[PendingKVPublication] = []
        try:
            for request_id, state, advance in zip(request_ids, states, advances, strict=True):
                target_length = state.sequence_length + advance
                pages_needed = (target_length + self.page_size - 1) // self.page_size
                missing = pages_needed - len(state.page_indices)
                if missing > len(self._free):
                    raise KVAllocationError(
                        f"Talker KV needs {missing} pages for {request_id!r}, only {len(self._free)} remain"
                    )
                new_pages = tuple(self._free[:missing])
                del self._free[:missing]
                allocated.extend(new_pages)
                proposed = (*state.page_indices, *new_pages)
                token = self._next_token
                self._next_token += 1
                pending = _Pending(token, state, proposed, new_pages, advance)
                self._pending[request_id] = pending
                created.append(request_id)
                publications.append(
                    PendingKVPublication(
                        self,
                        request_id,
                        token,
                        state.version,
                        proposed,
                        advance,
                    )
                )
        except Exception:
            for request_id in created:
                del self._pending[request_id]
            self._free.extend(allocated)
            self._free.sort()
            raise
        return KVReservation(tuple(publications))

    def _complete(self, publication: PendingKVPublication, *, publish: bool) -> None:
        pending = self._pending.get(publication.request_id)
        current = self._states.get(publication.request_id)
        if (
            pending is None
            or current is None
            or pending.status != "pending"
            or pending.token != publication.token
            or pending.prior.version != publication.prior_version
            or current != pending.prior
            or pending.proposed_pages != publication.page_indices
        ):
            raise StaleKVPublicationError("Talker KV publication is stale or no longer pending")
        if publish:
            self._states[publication.request_id] = RequestKVState(
                request_id=publication.request_id,
                page_indices=pending.proposed_pages,
                sequence_length=current.sequence_length + pending.sequence_advance,
                position_start=current.position_start + pending.sequence_advance,
                version=current.version + 1,
            )
            pending.status = "published"
        else:
            self._free.extend(pending.new_pages)
            self._free.sort()
            pending.status = "discarded"
        del self._pending[publication.request_id]


__all__ = [
    "KVAllocationError",
    "PendingKVPublication",
    "KVReservation",
    "PagedKVAllocator",
    "RequestKVState",
    "StaleKVPublicationError",
]

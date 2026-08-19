from __future__ import annotations

import pytest
import torch

from nari_qwen3_tts.executor import (
    CudaGraphPoolFence,
    KVAllocationError,
    PagedKVAllocator,
    SlotBusyError,
    SlotLeaseState,
    StaleKVPublicationError,
    TalkerCodebookAddress,
    stage_rows,
)


def test_static_staging_resets_padding_and_rejects_shape_dtype_drift() -> None:
    destination = torch.full((4, 3), 99, dtype=torch.int64)
    stage_rows(destination, torch.tensor([[1, 2, 3], [4, 5, 6]]), logical_rows=2)
    assert destination.tolist() == [[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]]
    stage_rows(destination, torch.tensor([[7, 8, 9]]), logical_rows=1)
    assert destination.tolist() == [[7, 8, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="dtype"):
        stage_rows(destination, torch.ones((1, 3), dtype=torch.float32), logical_rows=1)
    with pytest.raises(ValueError, match="shape"):
        stage_rows(destination, torch.ones((1, 4), dtype=torch.int64), logical_rows=1)


@pytest.mark.parametrize("logical_rows", [True, False, 1.0, "1"])
def test_static_staging_rejects_non_integer_logical_row_counts(logical_rows: object) -> None:
    destination = torch.full((2, 1), 9, dtype=torch.int64)
    source = torch.ones((1, 1), dtype=torch.int64)
    before = destination.clone()
    with pytest.raises((TypeError, ValueError)):
        stage_rows(destination, source, logical_rows=logical_rows)  # type: ignore[arg-type]
    assert torch.equal(destination, before)


def test_static_slots_have_generation_checked_exclusive_leases() -> None:
    lease_state = SlotLeaseState()
    first = lease_state.reserve()
    with pytest.raises(SlotBusyError, match="in flight"):
        lease_state.reserve()
    lease_state.release(first)
    replacement = lease_state.reserve()
    assert replacement.generation != first.generation
    with pytest.raises(SlotBusyError, match="stale"):
        lease_state.release(first)
    lease_state.release(replacement)
    assert lease_state.available
    with pytest.raises(SlotBusyError, match="stale"):
        lease_state.release(replacement)


def test_submission_fence_fails_closed_while_shared_graph_memory_is_in_flight() -> None:
    fence = CudaGraphPoolFence(device=torch.device("cpu"))
    first = fence.reserve()
    with pytest.raises(SlotBusyError, match="CUDA Graph memory pool"):
        fence.reserve()
    fence.release(first)
    successor = fence.reserve()
    assert successor.generation > first.generation
    fence.release(successor)


def test_submission_fence_rejects_stale_release() -> None:
    fence = CudaGraphPoolFence(device=torch.device("cpu"))
    lease = fence.reserve()
    fence.release(lease)
    with pytest.raises(SlotBusyError, match="stale or unowned"):
        fence.release(lease)


def test_paged_kv_publication_is_pending_stale_safe_and_discardable() -> None:
    allocator = PagedKVAllocator(total_pages=12, page_size=4)
    allocator.add_request("a")
    reservation = allocator.reserve(("a",), (5,))
    publication = reservation.publications[0]
    assert allocator.state("a").sequence_length == 0
    assert len(publication.page_indices) == 2
    publication.publish()
    assert allocator.state("a").sequence_length == 5
    assert allocator.state("a").version == 1
    with pytest.raises(StaleKVPublicationError):
        publication.publish()

    discarded = allocator.reserve(("a",), (1,)).publications[0]
    discarded.discard()
    assert allocator.state("a").sequence_length == 5
    with pytest.raises(StaleKVPublicationError):
        discarded.publish()


def test_paged_kv_rejects_duplicate_rows_and_inflight_aliasing() -> None:
    allocator = PagedKVAllocator(total_pages=12, page_size=4)
    allocator.add_request("a")
    allocator.add_request("b")
    with pytest.raises(ValueError, match="unique"):
        allocator.reserve(("a", "a"), (1, 1))
    reservation = allocator.reserve(("a", "b"), (1, 2))
    with pytest.raises(SlotBusyError, match="in flight"):
        allocator.reserve(("a",), (1,))
    for publication in reservation.publications:
        publication.discard()


def test_paged_kv_failed_multirow_reservation_rolls_back_every_page_and_owner() -> None:
    allocator = PagedKVAllocator(total_pages=3, page_size=4)
    allocator.add_request("a")
    allocator.add_request("b")

    with pytest.raises(KVAllocationError):
        allocator.reserve(("a", "b"), (1, 9))

    assert allocator.free_pages == 3
    assert allocator.state("a").sequence_length == 0
    assert allocator.state("b").sequence_length == 0
    # A failed cohort must not leave either request marked in-flight.
    successor = allocator.reserve(("b",), (9,))
    successor.publications[0].discard()
    assert allocator.free_pages == 3


def test_paged_kv_recycles_capacity_across_more_requests_than_the_page_pool() -> None:
    allocator = PagedKVAllocator(total_pages=3, page_size=4)

    for index in range(12):
        request_id = f"request-{index}"
        allocator.add_request(request_id)
        publication = allocator.reserve((request_id,), (9,)).publications[0]
        publication.publish()
        assert allocator.free_pages == 0
        allocator.remove_request(request_id)
        assert allocator.free_pages == 3


def test_talker_and_code_predictor_share_one_nonoverlapping_logical_rng_space() -> None:
    addresses = [
        TalkerCodebookAddress(frame_index=frame, codebook_index=codebook)
        for frame in range(3)
        for codebook in range(16)
    ]
    offsets = [address.offset for address in addresses]
    assert len(set(offsets)) == len(offsets)
    assert TalkerCodebookAddress(2, 0).is_talker
    assert not TalkerCodebookAddress(2, 1).is_talker
    assert TalkerCodebookAddress(2, 15).offset < TalkerCodebookAddress(3, 0).offset
    with pytest.raises(ValueError):
        TalkerCodebookAddress(0, 16)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_codebooks": 8},
        {"philox_stride": 16},
        {"num_codebooks": 32, "philox_stride": 64},
    ],
)
def test_rng_address_space_cannot_be_reconfigured_away_from_qwen3_tts(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="fixed.*16.*32"):
        TalkerCodebookAddress(frame_index=0, codebook_index=0, **kwargs)

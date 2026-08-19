from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts.contract import TalkerDecodeCaptureKey, TalkerPrefillCaptureKey
from nari_qwen3_tts.executor import PagedTalkerKV
from nari_qwen3_tts.executor.talker_kv import (
    FlashInferDecodeBinding,
    FlashInferPrefillBinding,
    TalkerAttentionMetadata,
)


def test_prefill_metadata_stages_request_pages_positions_writes_and_padding() -> None:
    cache = PagedTalkerKV(
        num_layers=2,
        num_kv_heads=2,
        num_qo_heads=4,
        head_dim=8,
        total_pages=16,
        page_size=4,
        scratch_page_count=4,
        workspace_bytes=1024,
        device=torch.device("cpu"),
        dtype=torch.float32,
        binding_factory=lambda **kwargs: kwargs,
    )
    cache.add_request("a")
    cache.add_request("b")
    key = TalkerPrefillCaptureKey(capture_batch_size=4, token_capacity=8, capture_sequence_length=None)
    context = cache.create_prefill(key, slot=0)
    publications = cache.prepare_prefill(context, key, ("a", "b"), (3, 2))
    metadata = context["metadata"]

    assert metadata.qo_indptr.tolist() == [0, 3, 5, 5, 5]
    assert metadata.position_ids.tolist() == [0, 1, 2, 0, 1, 0, 0, 0]
    assert metadata.write_pages[:5].tolist() == [
        publications[0].page_indices[0],
        publications[0].page_indices[0],
        publications[0].page_indices[0],
        publications[1].page_indices[0],
        publications[1].page_indices[0],
    ]
    assert metadata.write_offsets[:5].tolist() == [0, 1, 2, 0, 1]
    assert len(metadata.paged_kv_last_page_len) == 4
    assert len(set(metadata.padding_pages)) == 4
    assert len(set(metadata.paged_kv_indices[-2:].tolist())) == 2
    padding_writes = set(zip(metadata.write_pages[5:].tolist(), metadata.write_offsets[5:].tolist(), strict=True))
    assert len(padding_writes) == 3
    assert not padding_writes & set(
        zip(metadata.write_pages[:5].tolist(), metadata.write_offsets[:5].tolist(), strict=True)
    )
    cache.abort(publications)


def test_decode_metadata_uses_committed_lengths_and_proposed_next_write() -> None:
    cache = PagedTalkerKV(
        num_layers=2,
        num_kv_heads=2,
        num_qo_heads=4,
        head_dim=8,
        total_pages=16,
        page_size=4,
        scratch_page_count=2,
        workspace_bytes=1024,
        device=torch.device("cpu"),
        dtype=torch.float32,
        binding_factory=lambda **kwargs: kwargs,
    )
    cache.add_request("a")
    prefill_key = TalkerPrefillCaptureKey(1, 4, None)
    prefill = cache.create_prefill(prefill_key, slot=0)
    initial = cache.prepare_prefill(prefill, prefill_key, ("a",), (3,))
    initial[0].publish()

    key = TalkerDecodeCaptureKey(2)
    context = cache.create_decode(key, slot=0)
    publications = cache.prepare_decode(context, key, ("a",))
    metadata = context["metadata"]
    assert metadata.position_ids.tolist() == [3, 0]
    assert metadata.write_pages[0].item() == publications[0].page_indices[0]
    assert metadata.write_offsets[0].item() == 3
    assert metadata.write_pages[1].item() == metadata.padding_pages[1]
    assert cache.state("a").sequence_length == 3
    cache.abort(publications)
    assert cache.state("a").sequence_length == 3


def test_metadata_rejects_capture_larger_than_private_scratch_capacity() -> None:
    cache = PagedTalkerKV(
        num_layers=2,
        num_kv_heads=2,
        num_qo_heads=4,
        head_dim=8,
        total_pages=16,
        page_size=4,
        scratch_page_count=1,
        workspace_bytes=1024,
        device=torch.device("cpu"),
        dtype=torch.float32,
        binding_factory=lambda **kwargs: kwargs,
    )
    cache.add_request("a")

    decode_key = TalkerDecodeCaptureKey(2)
    with pytest.raises(ValueError, match="CUDA Graph rows exceed"):
        cache.prepare_decode(cache.create_decode(decode_key, slot=0), decode_key, ("a",))

    prefill_key = TalkerPrefillCaptureKey(1, 8, None)
    with pytest.raises(ValueError, match="token capacity exceeds"):
        cache.prepare_prefill(
            cache.create_prefill(prefill_key, slot=1),
            prefill_key,
            ("a",),
            (1,),
        )


def test_decode_attention_plan_reuses_page_count_shape_while_refreshing_dynamic_lengths() -> None:
    class Wrapper:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, **kwargs) -> None:
            self.calls.append(kwargs)

    binding = object.__new__(FlashInferDecodeBinding)
    binding.capture_batch_size = 2
    binding.num_qo_heads = 4
    binding.num_kv_heads = 2
    binding.head_dim = 8
    binding.page_size = 4
    binding.dtype = torch.float32
    binding.position_ids = torch.zeros(2, dtype=torch.long)
    binding.write_pages = torch.zeros(2, dtype=torch.long)
    binding.write_offsets = torch.zeros(2, dtype=torch.long)
    binding.paged_kv_indptr = torch.zeros(3, dtype=torch.int32)
    binding.paged_kv_indices = torch.zeros(8, dtype=torch.int32)
    binding.paged_kv_last_page_len = torch.ones(2, dtype=torch.int32)
    binding.metadata = None
    binding.wrapper = Wrapper()
    binding._cuda_graph_plan_signature = None

    def metadata(*, indptr, indices, last_page_len):
        return TalkerAttentionMetadata(
            qo_indptr=None,
            paged_kv_indptr=torch.tensor(indptr, dtype=torch.int32),
            paged_kv_indices=torch.tensor(indices, dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
            position_ids=torch.tensor([3, 5]),
            write_pages=torch.tensor([1, 2]),
            write_offsets=torch.tensor([3, 1]),
            padding_pages=(),
        )

    binding.plan(metadata(indptr=[0, 1, 3], indices=[1, 2, 3], last_page_len=[4, 2]))
    binding.plan(metadata(indptr=[0, 1, 3], indices=[4, 5, 6], last_page_len=[4, 2]))

    assert len(binding.wrapper.calls) == 1
    assert binding.paged_kv_indices[:3].tolist() == [4, 5, 6]
    assert binding.paged_kv_last_page_len.tolist() == [4, 2]

    # The captured decode kernel plan depends on physical row page counts. Page
    # IDs and exact final-page lengths remain dynamic persistent-buffer data.
    binding.plan(metadata(indptr=[0, 1, 3], indices=[4, 5, 6], last_page_len=[1, 3]))
    assert len(binding.wrapper.calls) == 1
    assert binding.paged_kv_last_page_len.tolist() == [1, 3]

    binding.plan(metadata(indptr=[0, 2, 3], indices=[4, 5, 6], last_page_len=[4, 1]))
    assert len(binding.wrapper.calls) == 2


def test_prefill_attention_plan_reuses_shape_and_refreshes_request_page_metadata() -> None:
    class Wrapper:
        def __init__(self) -> None:
            self.calls = []

        def plan(self, **kwargs) -> None:
            self.calls.append(kwargs)

    binding = object.__new__(FlashInferPrefillBinding)
    binding.capture_batch_size = 2
    binding.num_qo_heads = 4
    binding.num_kv_heads = 2
    binding.head_dim = 8
    binding.page_size = 4
    binding.dtype = torch.float32
    binding.position_ids = torch.zeros(4, dtype=torch.long)
    binding.write_pages = torch.zeros(4, dtype=torch.long)
    binding.write_offsets = torch.zeros(4, dtype=torch.long)
    binding.qo_indptr = torch.zeros(3, dtype=torch.int32)
    binding.paged_kv_indptr = torch.zeros(3, dtype=torch.int32)
    binding.paged_kv_indices = torch.zeros(8, dtype=torch.int32)
    binding.paged_kv_last_page_len = torch.ones(2, dtype=torch.int32)
    binding.metadata = None
    binding.wrapper = Wrapper()
    binding._cuda_graph_plan_signature = None

    def metadata(*, qo_indptr, indptr, indices, last_page_len):
        return TalkerAttentionMetadata(
            qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32),
            paged_kv_indptr=torch.tensor(indptr, dtype=torch.int32),
            paged_kv_indices=torch.tensor(indices, dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
            position_ids=torch.tensor([0, 1, 0, 0]),
            write_pages=torch.tensor([1, 1, 2, 7]),
            write_offsets=torch.tensor([0, 1, 0, 0]),
            padding_pages=(7,),
        )

    first = metadata(
        qo_indptr=[0, 2, 3],
        indptr=[0, 1, 2],
        indices=[1, 2],
        last_page_len=[2, 1],
    )
    binding.plan(first)
    second = metadata(
        qo_indptr=[0, 2, 3],
        indptr=[0, 1, 2],
        indices=[4, 5],
        last_page_len=[2, 1],
    )
    binding.plan(second)

    # Request page identities vary every decision, but an unchanged packed
    # query/page-count shape must not pay another FlashInfer plan().
    assert len(binding.wrapper.calls) == 1
    assert binding.qo_indptr.tolist() == [0, 2, 3]
    assert binding.paged_kv_indptr.tolist() == [0, 1, 2]
    assert binding.paged_kv_indices[:2].tolist() == [4, 5]
    assert binding.paged_kv_last_page_len.tolist() == [2, 1]

    changed_last_page_lengths = metadata(
        qo_indptr=[0, 2, 3],
        indptr=[0, 1, 2],
        indices=[4, 5],
        last_page_len=[1, 1],
    )
    binding.plan(changed_last_page_lengths)
    assert len(binding.wrapper.calls) == 2

    changed_query_shape = metadata(
        qo_indptr=[0, 1, 3],
        indptr=[0, 1, 2],
        indices=[4, 5],
        last_page_len=[1, 2],
    )
    binding.plan(changed_query_shape)
    assert len(binding.wrapper.calls) == 3

    changed_page_count_shape = metadata(
        qo_indptr=[0, 1, 3],
        indptr=[0, 2, 3],
        indices=[4, 5, 6],
        last_page_len=[1, 2],
    )
    binding.plan(changed_page_count_shape)
    assert len(binding.wrapper.calls) == 4


def test_attention_plan_reuse_is_invalidated_when_a_shared_workspace_changes_owner() -> None:
    """A different capture binding may overwrite shared FlashInfer plan workspace."""

    class Wrapper:
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, **kwargs) -> None:
            del kwargs
            self.calls += 1

    def binding(shared) -> FlashInferDecodeBinding:
        value = object.__new__(FlashInferDecodeBinding)
        value.capture_batch_size = 1
        value.num_qo_heads = 4
        value.num_kv_heads = 2
        value.head_dim = 8
        value.page_size = 4
        value.dtype = torch.float32
        value.position_ids = torch.zeros(1, dtype=torch.long)
        value.write_pages = torch.zeros(1, dtype=torch.long)
        value.write_offsets = torch.zeros(1, dtype=torch.long)
        value.paged_kv_indptr = torch.zeros(2, dtype=torch.int32)
        value.paged_kv_indices = torch.zeros(4, dtype=torch.int32)
        value.paged_kv_last_page_len = torch.ones(1, dtype=torch.int32)
        value.metadata = None
        value.wrapper = Wrapper()
        value._cuda_graph_plan_signature = None
        value._shared_plan_state = shared
        return value

    metadata = TalkerAttentionMetadata(
        qo_indptr=None,
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        position_ids=torch.tensor([0]),
        write_pages=torch.tensor([1]),
        write_offsets=torch.tensor([0]),
        padding_pages=(),
    )
    shared = SimpleNamespace(owner=None)
    first = binding(shared)
    second = binding(shared)

    first.plan(metadata)
    second.plan(metadata)
    first.plan(metadata)

    assert first.wrapper.calls == 2
    assert second.wrapper.calls == 1

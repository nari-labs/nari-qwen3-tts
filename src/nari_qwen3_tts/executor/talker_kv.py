"""Paged Talker KV and direct FlashInfer CUDA Graph bindings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import torch

from nari_qwen3_tts.contract.stage import TalkerDecodeCaptureKey, TalkerPrefillCaptureKey
from nari_qwen3_tts.executor.cache import PagedKVAllocator, PendingKVPublication, RequestKVState


@dataclass(frozen=True, slots=True)
class TalkerAttentionMetadata:
    qo_indptr: torch.Tensor | None
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    position_ids: torch.Tensor
    write_pages: torch.Tensor
    write_offsets: torch.Tensor
    padding_pages: tuple[int, ...]
    reuse_attention_plan: bool = True


class TalkerAttentionContext(Protocol):
    metadata: TalkerAttentionMetadata | None

    def plan(self, metadata: TalkerAttentionMetadata) -> None: ...

    def select_layer(self, index: int) -> None: ...

    def attend(
        self,
        attention: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor: ...

    def advance_sequence_lengths(self) -> None: ...


@dataclass(slots=True)
class _SharedPlanState:
    """Binding whose FlashInfer plan currently owns one shared workspace."""

    active_binding: object | None = None


class _FlashInferBindingBase:
    def __init__(
        self,
        *,
        storage: torch.Tensor,
        capture_batch_size: int,
        token_capacity: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        total_pages: int,
        workspace: torch.Tensor,
        shared_plan_state: _SharedPlanState,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.storage = storage
        self.capture_batch_size = capture_batch_size
        self.token_capacity = token_capacity
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.total_pages = total_pages
        self.device = device
        self.dtype = dtype
        if workspace.dtype is not torch.uint8 or workspace.device != device:
            raise ValueError("Talker FlashInfer workspace must be a uint8 tensor on the execution device")
        self.workspace = workspace
        self._shared_plan_state = shared_plan_state
        self.position_ids = torch.zeros(token_capacity, dtype=torch.long, device=device)
        self.write_pages = torch.zeros(token_capacity, dtype=torch.long, device=device)
        self.write_offsets = torch.zeros(token_capacity, dtype=torch.long, device=device)
        self.metadata: TalkerAttentionMetadata | None = None
        self._layer_index = 0

    def _stage(self, metadata: TalkerAttentionMetadata) -> None:
        if metadata.position_ids.shape != self.position_ids.shape:
            raise ValueError("Talker position metadata does not match the captured token capacity")
        non_blocking = metadata.position_ids.device.type == "cpu" and metadata.position_ids.is_pinned()
        self.position_ids.copy_(metadata.position_ids, non_blocking=non_blocking)
        self.write_pages.copy_(metadata.write_pages, non_blocking=non_blocking)
        self.write_offsets.copy_(metadata.write_offsets, non_blocking=non_blocking)
        self.metadata = metadata

    def select_layer(self, index: int) -> None:
        if not 0 <= index < self.storage.shape[0]:
            raise IndexError("Talker layer index exceeds paged KV storage")
        self._layer_index = index

    def attend(
        self,
        attention: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # Imported here, not at module scope: the kernel module requires Triton,
        # and the CPU contract tests exercise everything up to this call.
        from nari_qwen3_tts.model.kernels import paired_qk_norm_rope_paged_cache_

        layer = self.storage[self._layer_index]
        query, key = paired_qk_norm_rope_paged_cache_(
            query,
            key,
            value,
            attention.q_norm.weight,
            attention.k_norm.weight,
            self.position_ids,
            attention.q_norm.variance_epsilon,
            float(attention.rope_theta),
            layer,
            self.write_pages,
            self.write_offsets,
        )
        return self._run(query, layer).to(query.dtype)

    def _run(self, query: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def advance_sequence_lengths() -> None:
        # Replay proposes a typed KV publication. The Engine owns committed length.
        return None


class FlashInferPrefillBinding(_FlashInferBindingBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        import flashinfer

        self.qo_indptr = torch.zeros(self.capture_batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.zeros(self.capture_batch_size + 1, dtype=torch.int32, device=self.device)
        # Padding rows may repeat the scratch page, so metadata entries can exceed
        # the count of unique storage pages without exceeding this static buffer.
        self.paged_kv_indices = torch.zeros(
            self.total_pages + self.capture_batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.paged_kv_last_page_len = torch.ones(
            self.capture_batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self.qo_indptr,
            paged_kv_indptr_buf=self.paged_kv_indptr,
            paged_kv_indices_buf=self.paged_kv_indices,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len,
        )
        self._cuda_graph_plan_signature: tuple[
            torch.dtype,
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ] | None = None

    def plan(self, metadata: TalkerAttentionMetadata) -> None:
        if metadata.qo_indptr is None:
            raise ValueError("Talker prefill requires packed query metadata")
        if metadata.paged_kv_indices.numel() > self.paged_kv_indices.numel():
            raise ValueError("Talker page metadata exceeds its captured buffer")
        self._stage(metadata)
        if metadata.qo_indptr.device.type != "cpu" or metadata.paged_kv_indptr.device.type != "cpu":
            raise ValueError("Talker prefill page-count layout must be staged from CPU metadata")
        query_indptr = metadata.qo_indptr.tolist()
        page_indptr = metadata.paged_kv_indptr.tolist()
        signature = (
            self.dtype,
            tuple(stop - start for start, stop in pairwise(query_indptr)),
            tuple(stop - start for start, stop in pairwise(page_indptr)),
            tuple(metadata.paged_kv_last_page_len.tolist()),
        )
        self.qo_indptr.copy_(metadata.qo_indptr, non_blocking=True)
        self.paged_kv_indptr.copy_(metadata.paged_kv_indptr, non_blocking=True)
        self.paged_kv_indices.zero_()
        self.paged_kv_indices[: metadata.paged_kv_indices.numel()].copy_(
            metadata.paged_kv_indices,
            non_blocking=True,
        )
        self.paged_kv_last_page_len.copy_(
            metadata.paged_kv_last_page_len,
            non_blocking=True,
        )
        shared_plan_state = getattr(self, "_shared_plan_state", None)
        if signature == self._cuda_graph_plan_signature and (
            shared_plan_state is None or shared_plan_state.active_binding is self
        ):
            return
        if shared_plan_state is not None:
            shared_plan_state.active_binding = None
        self.wrapper.plan(
            qo_indptr=metadata.qo_indptr,
            paged_kv_indptr=metadata.paged_kv_indptr,
            paged_kv_indices=metadata.paged_kv_indices,
            paged_kv_last_page_len=metadata.paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=True,
            q_data_type=self.dtype,
        )
        self._cuda_graph_plan_signature = signature
        if shared_plan_state is not None:
            shared_plan_state.active_binding = self

    def _run(self, query: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        return self.wrapper.run(query.to(self.dtype), layer)


class FlashInferDecodeBinding(_FlashInferBindingBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        import flashinfer

        self.paged_kv_indptr = torch.zeros(self.capture_batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.zeros(
            self.total_pages + self.capture_batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.paged_kv_last_page_len = torch.ones(
            self.capture_batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=self.paged_kv_indptr,
            paged_kv_indices_buffer=self.paged_kv_indices,
            paged_kv_last_page_len_buffer=self.paged_kv_last_page_len,
        )
        self._cuda_graph_plan_signature: tuple[
            torch.dtype,
            tuple[int, ...],
        ] | None = None

    def plan(self, metadata: TalkerAttentionMetadata) -> None:
        if metadata.qo_indptr is not None:
            raise ValueError("Talker decode cannot contain packed query metadata")
        if metadata.paged_kv_indices.numel() > self.paged_kv_indices.numel():
            raise ValueError("Talker page metadata exceeds its captured buffer")
        self._stage(metadata)
        if metadata.paged_kv_indptr.device.type != "cpu":
            raise ValueError("Talker page-count layout must be staged from CPU metadata")
        indptr = metadata.paged_kv_indptr.tolist()
        page_counts = tuple(stop - start for start, stop in pairwise(indptr))
        signature = (
            self.dtype,
            page_counts,
        )
        shared_plan_state = getattr(self, "_shared_plan_state", None)
        if metadata.reuse_attention_plan and signature == self._cuda_graph_plan_signature and (
            shared_plan_state is None or shared_plan_state.active_binding is self
        ):
            self.paged_kv_indptr.copy_(metadata.paged_kv_indptr, non_blocking=True)
            self.paged_kv_indices[: metadata.paged_kv_indices.numel()].copy_(
                metadata.paged_kv_indices,
                non_blocking=True,
            )
            self.paged_kv_last_page_len.copy_(
                metadata.paged_kv_last_page_len,
                non_blocking=True,
            )
            return
        if shared_plan_state is not None:
            shared_plan_state.active_binding = None
        self.wrapper.plan(
            indptr=metadata.paged_kv_indptr,
            indices=metadata.paged_kv_indices,
            last_page_len=metadata.paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            q_data_type=self.dtype,
        )
        self._cuda_graph_plan_signature = (
            signature if metadata.reuse_attention_plan else None
        )
        if shared_plan_state is not None:
            shared_plan_state.active_binding = self

    def _run(self, query: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        return self.wrapper.run(query.to(self.dtype), layer)


class PagedTalkerKV:
    """Request state, page metadata, and one direct binding per Talker CUDA Graph slot."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        num_qo_heads: int,
        head_dim: int,
        total_pages: int,
        page_size: int,
        scratch_page_count: int,
        workspace_bytes: int,
        device: torch.device,
        dtype: torch.dtype,
        binding_factory=None,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.total_pages = total_pages
        self.page_size = page_size
        self.workspace_bytes = workspace_bytes
        self.device = device
        self.dtype = dtype
        self.storage = torch.zeros(
            (num_layers, total_pages, 2, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        ).contiguous()
        self.allocator = PagedKVAllocator(total_pages=total_pages, page_size=page_size)
        self._scratch_pages = self.allocator.claim_scratch_pages(scratch_page_count)
        self._binding_factory = binding_factory
        self._next_capture = 0
        self._capture_requests: dict[int, tuple[str, ...]] = {}
        self._workspaces: dict[int, torch.Tensor] = {}
        self._workspace_plan_states: dict[int, _SharedPlanState] = {}

    def add_request(self, request_id: str) -> None:
        self.allocator.add_request(request_id)

    def remove_request(self, request_id: str) -> None:
        self.allocator.remove_request(request_id)

    def state(self, request_id: str) -> RequestKVState:
        return self.allocator.state(request_id)

    def _binding(self, key: TalkerPrefillCaptureKey | TalkerDecodeCaptureKey, slot: int):
        prefill = isinstance(key, TalkerPrefillCaptureKey)
        factory = self._binding_factory or (FlashInferPrefillBinding if prefill else FlashInferDecodeBinding)
        workspace = self._workspaces.get(slot)
        if workspace is None:
            workspace = torch.empty(self.workspace_bytes, dtype=torch.uint8, device=self.device)
            self._workspaces[slot] = workspace
        shared_plan_state = self._workspace_plan_states.setdefault(slot, _SharedPlanState())
        arguments = dict(
            storage=self.storage,
            capture_batch_size=key.capture_batch_size,
            token_capacity=key.token_capacity if prefill else key.capture_batch_size,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            total_pages=self.total_pages,
            workspace=workspace,
            shared_plan_state=shared_plan_state,
            device=self.device,
            dtype=self.dtype,
        )
        if self._binding_factory is not None:
            arguments.update(slot=slot, metadata=None)
        return factory(**arguments)

    def create_prefill(self, key: TalkerPrefillCaptureKey, *, slot: int):
        return self._binding(key, slot)

    def create_decode(self, key: TalkerDecodeCaptureKey, *, slot: int):
        return self._binding(key, slot)

    @staticmethod
    def _set_metadata(context: object, metadata: TalkerAttentionMetadata) -> None:
        if isinstance(context, dict):
            context["metadata"] = metadata
        else:
            context.plan(metadata)

    def _validate_scratch_capacity(self, *, capture_rows: int, token_capacity: int) -> None:
        if capture_rows > len(self._scratch_pages):
            raise ValueError("Talker CUDA Graph rows exceed executor-private scratch pages")
        if token_capacity > len(self._scratch_pages) * self.page_size:
            raise ValueError("Talker token capacity exceeds executor-private scratch slots")

    def _metadata(
        self,
        *,
        request_ids: tuple[str, ...],
        advances: tuple[int, ...],
        capture_rows: int,
        token_capacity: int,
        prefill: bool,
        reuse_attention_plan: bool = True,
    ) -> tuple[tuple[PendingKVPublication, ...], TalkerAttentionMetadata]:
        self._validate_scratch_capacity(capture_rows=capture_rows, token_capacity=token_capacity)
        reservation = self.allocator.reserve(request_ids, advances)
        publications = reservation.publications
        try:
            indptr = [0]
            page_indices: list[int] = []
            last_page_lengths: list[int] = []
            positions: list[int] = []
            write_pages: list[int] = []
            write_offsets: list[int] = []
            qo_indptr = [0]
            for row in range(capture_rows):
                if row < len(request_ids):
                    request_id = request_ids[row]
                    prior = self.state(request_id)
                    publication = publications[row]
                    pages = publication.page_indices
                    advance = advances[row]
                    total = prior.sequence_length + advance
                    page_indices.extend(pages)
                    indptr.append(len(page_indices))
                    last_page_lengths.append(total % self.page_size or self.page_size)
                    for offset in range(advance):
                        absolute = prior.sequence_length + offset
                        write_pages.append(pages[absolute // self.page_size])
                        write_offsets.append(absolute % self.page_size)
                        positions.append(prior.position_start + offset)
                    if prefill:
                        qo_indptr.append(qo_indptr[-1] + advance)
                else:
                    scratch = self._scratch_pages[row]
                    page_indices.append(scratch)
                    indptr.append(len(page_indices))
                    last_page_lengths.append(1)
                    if prefill:
                        qo_indptr.append(qo_indptr[-1])
                    else:
                        write_pages.append(scratch)
                        write_offsets.append(0)
                        positions.append(0)
            while len(positions) < token_capacity:
                scratch_slot = len(positions) - sum(advances)
                positions.append(0)
                write_pages.append(self._scratch_pages[scratch_slot // self.page_size])
                write_offsets.append(scratch_slot % self.page_size)
            if len(positions) != token_capacity:
                raise ValueError("Talker KV metadata exceeds the captured token capacity")
            pin_memory = self.device.type == "cuda"
            metadata = TalkerAttentionMetadata(
                qo_indptr=(
                    None
                    if not prefill
                    else torch.tensor(qo_indptr, dtype=torch.int32, pin_memory=pin_memory)
                ),
                paged_kv_indptr=torch.tensor(indptr, dtype=torch.int32, pin_memory=pin_memory),
                paged_kv_indices=torch.tensor(page_indices, dtype=torch.int32, pin_memory=pin_memory),
                paged_kv_last_page_len=torch.tensor(
                    last_page_lengths,
                    dtype=torch.int32,
                    pin_memory=pin_memory,
                ),
                position_ids=torch.tensor(positions, dtype=torch.long, pin_memory=pin_memory),
                write_pages=torch.tensor(write_pages, dtype=torch.long, pin_memory=pin_memory),
                write_offsets=torch.tensor(write_offsets, dtype=torch.long, pin_memory=pin_memory),
                padding_pages=self._scratch_pages,
                reuse_attention_plan=reuse_attention_plan,
            )
            return publications, metadata
        except Exception:
            self.abort(publications)
            raise

    def prepare_prefill(
        self,
        context: object,
        key: TalkerPrefillCaptureKey,
        request_ids: tuple[str, ...],
        sequence_lengths: tuple[int, ...],
    ) -> tuple[PendingKVPublication, ...]:
        if len(request_ids) != len(sequence_lengths) or len(request_ids) > key.capture_batch_size:
            raise ValueError("Talker prefill request rows do not fit the capture")
        if sum(sequence_lengths) > key.token_capacity:
            raise ValueError("Talker prefill tokens exceed the capture")
        publications, metadata = self._metadata(
            request_ids=request_ids,
            advances=sequence_lengths,
            capture_rows=key.capture_batch_size,
            token_capacity=key.token_capacity,
            prefill=True,
        )
        try:
            self._set_metadata(context, metadata)
        except Exception:
            self.abort(publications)
            raise
        return publications

    def prepare_decode(
        self,
        context: object,
        key: TalkerDecodeCaptureKey,
        request_ids: tuple[str, ...],
        *,
        reuse_attention_plan: bool = True,
    ) -> tuple[PendingKVPublication, ...]:
        if not request_ids or len(request_ids) > key.capture_batch_size:
            raise ValueError("Talker decode request rows do not fit the capture")
        publications, metadata = self._metadata(
            request_ids=request_ids,
            advances=(1,) * len(request_ids),
            capture_rows=key.capture_batch_size,
            token_capacity=key.capture_batch_size,
            prefill=False,
            reuse_attention_plan=reuse_attention_plan,
        )
        try:
            self._set_metadata(context, metadata)
        except Exception:
            self.abort(publications)
            raise
        return publications

    def prepare_capture(
        self,
        context: object,
        key: TalkerPrefillCaptureKey | TalkerDecodeCaptureKey,
    ) -> tuple[PendingKVPublication, ...]:
        capture_id = self._next_capture
        self._next_capture += 1
        request_ids = tuple(f"__cuda_capture_{capture_id}_{row}__" for row in range(key.capture_batch_size))
        for request_id in request_ids:
            self.add_request(request_id)
        self._capture_requests[id(context)] = request_ids
        if isinstance(key, TalkerDecodeCaptureKey):
            return self.prepare_decode(context, key, request_ids)
        base = key.token_capacity // key.capture_batch_size
        extra = key.token_capacity % key.capture_batch_size
        lengths = tuple(base + (1 if row < extra else 0) for row in range(key.capture_batch_size))
        return self.prepare_prefill(context, key, request_ids, lengths)

    def finish_capture(self, context: object, publications: tuple[PendingKVPublication, ...]) -> None:
        self.abort(publications)
        for request_id in self._capture_requests.pop(id(context), ()):
            self.remove_request(request_id)

    @staticmethod
    def abort(publications: tuple[PendingKVPublication, ...]) -> None:
        for publication in publications:
            publication.discard()


__all__ = [
    "FlashInferDecodeBinding",
    "FlashInferPrefillBinding",
    "PagedTalkerKV",
    "TalkerAttentionContext",
    "TalkerAttentionMetadata",
]

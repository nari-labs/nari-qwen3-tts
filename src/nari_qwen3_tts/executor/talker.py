"""Captured Talker prefill/decode with typed static staging and pending KV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import functional as F

from nari_qwen3_tts.contract.request import EncodedText
from nari_qwen3_tts.contract.stage import TalkerDecodeCaptureKey, TalkerPrefillCaptureKey
from nari_qwen3_tts.executor.cache import PendingKVPublication
from nari_qwen3_tts.executor.cuda_graph import (
    CapturedCall,
    CaptureDriver,
    CudaGraphPoolFence,
    SlotLeaseState,
    TorchCaptureDriver,
)
from nari_qwen3_tts.executor.input_layout import (
    TalkerInputPlan,
    prepare_talker_input_plan,
)
from nari_qwen3_tts.executor.rng import TALKER_FRAME_OFFSET_STRIDE
from nari_qwen3_tts.executor.rows import (
    TalkerDecodeRowsExecutionInput,
    TalkerExecutionResult,
    TalkerPrefillExecutionRow,
    TalkerPrefillRowsExecutionInput,
    TalkerSamplingExecutionRow,
)
from nari_qwen3_tts.executor.talker_kv import TalkerAttentionContext
from nari_qwen3_tts.executor.types import (
    TalkerDecodeInput,
    TalkerPrefillInput,
    TalkerResult,
    TalkerSamplingInput,
)


class TalkerKVBackend(Protocol):
    def create_prefill(self, key: TalkerPrefillCaptureKey, *, slot: int) -> TalkerAttentionContext: ...

    def create_decode(self, key: TalkerDecodeCaptureKey, *, slot: int) -> TalkerAttentionContext: ...

    def prepare_capture(
        self,
        context: TalkerAttentionContext,
        key: TalkerPrefillCaptureKey | TalkerDecodeCaptureKey,
    ) -> tuple[PendingKVPublication, ...] | None: ...

    def finish_capture(
        self,
        context: TalkerAttentionContext,
        publications: tuple[PendingKVPublication, ...],
    ) -> None: ...

    def prepare_prefill(
        self,
        context: TalkerAttentionContext,
        key: TalkerPrefillCaptureKey,
        request_ids: tuple[str, ...],
        sequence_lengths: tuple[int, ...],
    ) -> tuple[PendingKVPublication, ...]: ...

    def prepare_decode(
        self,
        context: TalkerAttentionContext,
        key: TalkerDecodeCaptureKey,
        request_ids: tuple[str, ...],
        *,
        reuse_attention_plan: bool = True,
    ) -> tuple[PendingKVPublication, ...]: ...

    def abort(self, publications: tuple[PendingKVPublication, ...]) -> None: ...


@dataclass(slots=True)
class _SamplingBuffers:
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    seed: torch.Tensor
    offsets: torch.Tensor
    seen: torch.Tensor


@dataclass(slots=True)
class _SamplingHostBuffers:
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    seed: torch.Tensor
    offsets: torch.Tensor


@dataclass(slots=True)
class _PrefillSlot:
    text: torch.Tensor
    codec: torch.Tensor
    mask: torch.Tensor
    last: torch.Tensor
    suppress: torch.Tensor
    sampling: _SamplingBuffers
    host_sampling: _SamplingHostBuffers
    host_text: torch.Tensor
    host_codec: torch.Tensor
    host_mask: torch.Tensor
    host_last: torch.Tensor
    host_suppress: torch.Tensor
    context: TalkerAttentionContext
    call: CapturedCall
    lease_state: SlotLeaseState


@dataclass(slots=True)
class _DecodeSlot:
    step_embed: torch.Tensor
    text: torch.Tensor
    last: torch.Tensor
    suppress: torch.Tensor
    sampling: _SamplingBuffers
    host_sampling: _SamplingHostBuffers
    host_text: torch.Tensor
    host_suppress: torch.Tensor
    context: TalkerAttentionContext
    call: CapturedCall
    lease_state: SlotLeaseState


class TalkerExecutor:
    """Stage-specific Talker CUDA Graphs; physical rows never own request state."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: object,
        cache: TalkerKVBackend,
        capture_slots: int,
        driver: CaptureDriver | None = None,
        submission_fence: CudaGraphPoolFence | None = None,
    ) -> None:
        if isinstance(capture_slots, bool) or not isinstance(capture_slots, int) or capture_slots < 1:
            raise ValueError("Talker capture slots must be a positive integer")
        self.model = model
        self.config = config
        self.cache = cache
        self.model.initialize_projected_text_embedding_cache()
        talker = self.config.talker
        self.hidden_size = int(talker.hidden_size)
        self.vocab_size = int(talker.vocab_size)
        self.device = torch.device(self.model.device)
        self.dtype = self.model.get_input_embeddings().weight.dtype
        self.capture_slots = capture_slots
        self.driver = driver or TorchCaptureDriver(device=self.device, autocast_dtype=self.dtype)
        self.submission_fence = submission_fence or CudaGraphPoolFence(device=self.device)
        self._suppress_mask: torch.Tensor | None = None
        self._prefill: dict[TalkerPrefillCaptureKey, tuple[_PrefillSlot, ...]] = {}
        self._decode: dict[TalkerDecodeCaptureKey, tuple[_DecodeSlot, ...]] = {}
        self._next_prefill: dict[TalkerPrefillCaptureKey, int] = {}
        self._next_decode: dict[TalkerDecodeCaptureKey, int] = {}

    def add_request(self, request_id: str) -> None:
        self.cache.add_request(request_id)

    def remove_request(self, request_id: str) -> None:
        self.cache.remove_request(request_id)

    def _get_suppress_mask(self) -> torch.Tensor:
        if self._suppress_mask is None:
            talker = self.config.talker
            codec_vocab_size = talker.code_predictor.vocab_size
            if not 0 < codec_vocab_size <= talker.vocab_size:
                raise ValueError("Codec vocabulary must fit inside the Talker vocabulary")
            mask = torch.zeros(talker.vocab_size, dtype=torch.bool, device=self.device)
            mask[codec_vocab_size:] = True
            mask[talker.codec_eos_token_id] = False
            self._suppress_mask = mask
        return self._suppress_mask

    def prepare_input_plan(
        self,
        *,
        text_inputs: list[torch.Tensor],
        instruct_inputs: list[torch.Tensor],
        languages: list[str],
        speakers: list[str],
        non_streaming_modes: list[bool],
    ) -> TalkerInputPlan:
        return prepare_talker_input_plan(
            self.config,
            text_inputs=text_inputs,
            instruct_inputs=instruct_inputs,
            languages=languages,
            speakers=speakers,
            non_streaming_modes=non_streaming_modes,
        )

    def prepare_prepared_inputs(self, prepared_inputs: list[EncodedText]) -> TalkerInputPlan:
        return self.prepare_input_plan(
            text_inputs=[prepared.text_token_ids for prepared in prepared_inputs],
            instruct_inputs=[prepared.instruct_token_ids for prepared in prepared_inputs],
            languages=[prepared.request.language for prepared in prepared_inputs],
            speakers=[prepared.request.voice for prepared in prepared_inputs],
            non_streaming_modes=[prepared.request.non_streaming_mode for prepared in prepared_inputs],
        )

    @staticmethod
    def _sample_direct(logits: torch.Tensor, values: TalkerSamplingInput) -> torch.Tensor:
        any_penalty = bool(torch.any(values.repetition_penalty != 1))
        from nari_qwen3_tts.model.sampling import sample_logits_stateless

        return sample_logits_stateless(
            logits=logits,
            temperature=values.temperature,
            top_k=values.top_k,
            top_p=values.top_p,
            repetition_penalty=values.repetition_penalty,
            seen_token_mask=values.seen_token_mask if any_penalty else None,
            any_greedy=bool(torch.any(values.temperature == 0)),
            any_top_k_zero=bool(torch.any(values.top_k == 0)),
            all_top_k_zero=bool(torch.all(values.top_k == 0)),
            seed=values.seed,
            offset=values.offsets,
        ).to(torch.long)

    def forward_logits(
        self,
        *,
        attention_context: TalkerAttentionContext,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model(input_embeds=input_embeds, attention_context=attention_context)
        last_hidden = hidden.index_select(0, last_token_indices)
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(self._get_suppress_mask().unsqueeze(0), -torch.inf)
        logits[:, self.config.talker.codec_eos_token_id].masked_fill_(suppress_eos, -torch.inf)
        return logits, last_hidden

    def materialize_prefill(
        self,
        text_token_ids: torch.Tensor,
        codec_token_ids: torch.Tensor,
        codec_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        text = F.embedding(text_token_ids, self.model.get_projected_text_embedding_cache())
        codec = F.embedding(codec_token_ids, self.model.get_input_embeddings().weight)
        return torch.where(codec_token_mask.unsqueeze(1), text + codec, text)

    def materialize_decode(
        self,
        talker_step_embed: torch.Tensor,
        text_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        text = F.embedding(text_token_ids, self.model.get_projected_text_embedding_cache())
        return talker_step_embed + text

    def _forward_direct(
        self,
        *,
        attention_context: TalkerAttentionContext,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
        sampling: TalkerSamplingInput,
    ) -> TalkerResult:
        logits, last_hidden = self.forward_logits(
            attention_context=attention_context,
            input_embeds=input_embeds,
            last_token_indices=last_token_indices,
            suppress_eos=suppress_eos,
        )
        return TalkerResult(
            tokens=self._sample_direct(logits, sampling),
            last_hidden=last_hidden,
            logits=logits,
        )

    def prefill(self, values: TalkerPrefillInput) -> TalkerResult:
        return self._forward_direct(
            attention_context=values.attention_context,
            input_embeds=self.materialize_prefill(
                values.text_token_ids,
                values.codec_token_ids,
                values.codec_token_mask,
            ),
            last_token_indices=values.last_token_indices,
            suppress_eos=values.suppress_eos,
            sampling=values.sampling,
        )

    def decode(self, values: TalkerDecodeInput) -> TalkerResult:
        input_embeds = self.materialize_decode(values.talker_step_embed, values.text_token_ids)
        return self._forward_direct(
            attention_context=values.attention_context,
            input_embeds=input_embeds,
            last_token_indices=torch.arange(values.sampling.rows, device=input_embeds.device),
            suppress_eos=values.suppress_eos,
            sampling=values.sampling,
        )

    @property
    def captured_cuda_graph_instances(self) -> int:
        return sum(len(slots) for slots in (*self._prefill.values(), *self._decode.values()))

    def _sampling(self, rows: int) -> _SamplingBuffers:
        return _SamplingBuffers(
            temperature=torch.zeros(rows, device=self.device),
            top_k=torch.ones(rows, dtype=torch.int32, device=self.device),
            top_p=torch.ones(rows, device=self.device),
            repetition_penalty=torch.ones(rows, device=self.device),
            seed=torch.zeros(rows, dtype=torch.long, device=self.device),
            offsets=torch.zeros(rows, dtype=torch.long, device=self.device),
            seen=torch.zeros((rows, self.vocab_size), dtype=torch.bool, device=self.device),
        )

    def _host_tensor(self, rows: int, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(
            rows,
            dtype=dtype,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )

    def _host_sampling(self, rows: int) -> _SamplingHostBuffers:
        return _SamplingHostBuffers(
            temperature=self._host_tensor(rows, dtype=torch.float32),
            top_k=self._host_tensor(rows, dtype=torch.int32),
            top_p=self._host_tensor(rows, dtype=torch.float32),
            repetition_penalty=self._host_tensor(rows, dtype=torch.float32),
            seed=self._host_tensor(rows, dtype=torch.long),
            offsets=self._host_tensor(rows, dtype=torch.long),
        )

    @staticmethod
    def _sample(logits: torch.Tensor, sampling: _SamplingBuffers) -> torch.Tensor:
        """One CUDA Graph-safe FlashInfer row-zero draw per logical physical row."""

        if not logits.is_cuda:
            from nari_qwen3_tts.model.sampling import sample_logits_stateless

            tokens = sample_logits_stateless(
                logits,
                sampling.temperature,
                sampling.top_k,
                sampling.top_p,
                sampling.seed,
                sampling.offsets,
                repetition_penalty=sampling.repetition_penalty,
                seen_token_mask=sampling.seen,
            )
            sampling.seen.scatter_(1, tokens.unsqueeze(1), True)
            return tokens

        from nari_qwen3_tts.executor.sampling import sample_talker_cuda_graph

        tokens = sample_talker_cuda_graph(
            logits,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.seed,
            sampling.offsets,
            sampling.repetition_penalty,
            sampling.seen,
        )
        sampling.seen.scatter_(1, tokens.unsqueeze(1), True)
        return tokens

    def capture(self, key: TalkerPrefillCaptureKey | TalkerDecodeCaptureKey) -> None:
        if isinstance(key, TalkerPrefillCaptureKey):
            self._capture_prefill(key)
        elif isinstance(key, TalkerDecodeCaptureKey):
            self._capture_decode(key)
        else:
            raise TypeError("Talker executor received the wrong capture key")

    def _finish_capture(
        self,
        context: TalkerAttentionContext,
        publications: tuple[PendingKVPublication, ...] | None,
    ) -> None:
        if publications is None:
            return
        finish = getattr(self.cache, "finish_capture", None)
        if finish is not None:
            finish(context, publications)
        else:
            self.cache.abort(publications)

    def _capture_prefill(self, key: TalkerPrefillCaptureKey) -> None:
        if key in self._prefill:
            return
        slots: list[_PrefillSlot] = []
        for slot_index in range(self.capture_slots):
            text = torch.zeros(key.token_capacity, dtype=torch.long, device=self.device)
            codec = torch.zeros_like(text)
            mask = torch.zeros(key.token_capacity, dtype=torch.bool, device=self.device)
            last = torch.zeros(key.capture_batch_size, dtype=torch.long, device=self.device)
            suppress = torch.ones(key.capture_batch_size, dtype=torch.bool, device=self.device)
            sampling = self._sampling(key.capture_batch_size)
            host_sampling = self._host_sampling(key.capture_batch_size)
            host_text = self._host_tensor(key.token_capacity, dtype=torch.long)
            host_codec = self._host_tensor(key.token_capacity, dtype=torch.long)
            host_mask = self._host_tensor(key.token_capacity, dtype=torch.bool)
            host_last = self._host_tensor(key.capture_batch_size, dtype=torch.long)
            host_suppress = self._host_tensor(key.capture_batch_size, dtype=torch.bool)
            context = self.cache.create_prefill(key, slot=slot_index)
            publications = self.cache.prepare_capture(context, key)

            def operation(
                text=text,
                codec=codec,
                mask=mask,
                last=last,
                suppress=suppress,
                sampling=sampling,
                context=context,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                embeds = self.materialize_prefill(text, codec, mask)
                logits, hidden = self.forward_logits(
                    attention_context=context,
                    input_embeds=embeds,
                    last_token_indices=last,
                    suppress_eos=suppress,
                )
                return self._sample(logits, sampling), hidden, logits

            try:
                call = self.driver.capture(operation)
            finally:
                self._finish_capture(context, publications)
            slots.append(
                _PrefillSlot(
                    text,
                    codec,
                    mask,
                    last,
                    suppress,
                    sampling,
                    host_sampling,
                    host_text,
                    host_codec,
                    host_mask,
                    host_last,
                    host_suppress,
                    context,
                    call,
                    SlotLeaseState(),
                )
            )
        self._prefill[key] = tuple(slots)
        self._next_prefill[key] = 0

    def _capture_decode(self, key: TalkerDecodeCaptureKey) -> None:
        if key in self._decode:
            return
        slots: list[_DecodeSlot] = []
        for slot_index in range(self.capture_slots):
            step = torch.zeros((key.capture_batch_size, self.hidden_size), dtype=self.dtype, device=self.device)
            text = torch.zeros(key.capture_batch_size, dtype=torch.long, device=self.device)
            last = torch.arange(key.capture_batch_size, dtype=torch.long, device=self.device)
            suppress = torch.ones(key.capture_batch_size, dtype=torch.bool, device=self.device)
            sampling = self._sampling(key.capture_batch_size)
            host_sampling = self._host_sampling(key.capture_batch_size)
            host_text = self._host_tensor(key.capture_batch_size, dtype=torch.long)
            host_suppress = self._host_tensor(key.capture_batch_size, dtype=torch.bool)
            context = self.cache.create_decode(key, slot=slot_index)
            publications = self.cache.prepare_capture(context, key)

            def operation(
                step=step,
                text=text,
                last=last,
                suppress=suppress,
                sampling=sampling,
                context=context,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                embeds = self.materialize_decode(step, text)
                logits, hidden = self.forward_logits(
                    attention_context=context,
                    input_embeds=embeds,
                    last_token_indices=last,
                    suppress_eos=suppress,
                )
                return self._sample(logits, sampling), hidden, logits

            try:
                call = self.driver.capture(operation)
            finally:
                self._finish_capture(context, publications)
            slots.append(
                _DecodeSlot(
                    step,
                    text,
                    last,
                    suppress,
                    sampling,
                    host_sampling,
                    host_text,
                    host_suppress,
                    context,
                    call,
                    SlotLeaseState(),
                )
            )
        self._decode[key] = tuple(slots)
        self._next_decode[key] = 0

    @staticmethod
    def _reserve(key, slots, next_slots):
        available = slots.get(key)
        if available is None:
            raise RuntimeError("Talker CUDA Graph has not been captured")
        index = next_slots[key]
        next_slots[key] = (index + 1) % len(available)
        slot = available[index]
        return slot, slot.lease_state.reserve()

    @staticmethod
    def _stage_sampling_rows(
        destination: _SamplingBuffers,
        host: _SamplingHostBuffers,
        sources: tuple[TalkerSamplingExecutionRow, ...],
    ) -> None:
        defaults = (
            (host.temperature, 0.0, "temperature"),
            (host.top_k, 1, "top_k"),
            (host.top_p, 1.0, "top_p"),
            (host.repetition_penalty, 1.0, "repetition_penalty"),
            (host.seed, 0, "seed"),
            (host.offsets, 0, "offset"),
        )
        device_values = (
            destination.temperature,
            destination.top_k,
            destination.top_p,
            destination.repetition_penalty,
            destination.seed,
            destination.offsets,
        )
        for (host_value, padding, name), device_value in zip(defaults, device_values, strict=True):
            host_value.fill_(padding)
            for row, source in enumerate(sources):
                host_value[row] = getattr(source, name)
            device_value.copy_(host_value, non_blocking=True)
        destination.seen.zero_()
        active_masks = tuple(
            (row, source.seen_token_mask)
            for row, source in enumerate(sources)
            if source.seen_token_mask is not None
        )
        if active_masks:
            torch._foreach_copy_(
                tuple(destination.seen[row] for row, _ in active_masks),
                tuple(mask for _, mask in active_masks),
            )

    def _validate_sampling_rows(
        self,
        sources: tuple[TalkerSamplingExecutionRow, ...],
    ) -> None:
        for source in sources:
            mask = source.seen_token_mask
            if mask is not None and mask.shape != (self.vocab_size,):
                raise ValueError("seen_token_mask must match the captured vocabulary")
            if mask is not None and mask.device != self.device:
                raise ValueError("seen_token_mask must be on the execution device")

    @staticmethod
    def _stack_rows(destination: torch.Tensor, sources: tuple[torch.Tensor, ...]) -> None:
        destination.zero_()
        if len(sources) == 1:
            destination[0].copy_(sources[0].reshape(destination.shape[1:]))
        elif sources:
            torch.stack(
                tuple(source.reshape(destination.shape[1:]) for source in sources),
                out=destination[: len(sources)],
            )

    def _result(
        self,
        *,
        slot: _PrefillSlot | _DecodeSlot,
        rows: int,
        publications: tuple[PendingKVPublication, ...],
        source_offsets: torch.Tensor,
    ) -> TalkerExecutionResult:
        captured = slot.call.replay()
        if not isinstance(captured, tuple) or len(captured) != 3:
            raise RuntimeError("Talker CUDA Graph returned an invalid result")
        tokens, hidden, logits = captured
        return TalkerExecutionResult(
            result=TalkerResult(
                tokens=tokens[:rows].clone(),
                last_hidden=hidden[:rows].clone(),
                logits=logits[:rows].clone(),
            ),
            next_seen_token_masks=slot.sampling.seen[:rows].clone(),
            next_sampling_offsets=source_offsets + TALKER_FRAME_OFFSET_STRIDE,
            kv_publications=publications,
        )

    def _stage_prefill_token_rows(
        self,
        slot: _PrefillSlot,
        rows: tuple[TalkerPrefillExecutionRow, ...],
        *,
        tokens: int,
    ) -> None:
        source_device = rows[0].text_token_ids.device
        if any(row.text_token_ids.device != source_device for row in rows):
            raise ValueError("Talker prefill rows must share one source device")
        if source_device.type != "cpu" and source_device != self.device:
            raise ValueError("Talker prefill rows must be on the host or execution device")
        sources = (
            ("text_token_ids", slot.text, slot.host_text),
            ("codec_token_ids", slot.codec, slot.host_codec),
            ("codec_token_mask", slot.mask, slot.host_mask),
        )
        if source_device.type == "cpu":
            for field, destination, host in sources:
                host.zero_()
                values = tuple(getattr(row, field) for row in rows)
                if len(values) == 1:
                    host[:tokens].copy_(values[0])
                else:
                    torch.cat(values, out=host[:tokens])
                destination.copy_(host, non_blocking=self.device.type == "cuda")
            return
        for field, destination, _host in sources:
            destination.zero_()
            values = tuple(getattr(row, field) for row in rows)
            if len(values) == 1:
                destination[:tokens].copy_(values[0])
            else:
                torch.cat(values, out=destination[:tokens])

    def replay(
        self,
        key: TalkerPrefillCaptureKey | TalkerDecodeCaptureKey,
        values: TalkerPrefillRowsExecutionInput | TalkerDecodeRowsExecutionInput,
    ) -> TalkerExecutionResult:
        submission = self.submission_fence.reserve()
        try:
            if isinstance(key, TalkerPrefillCaptureKey):
                return self._replay_prefill(key, values)
            if isinstance(key, TalkerDecodeCaptureKey):
                return self._replay_decode(key, values)
            raise TypeError("Talker executor received the wrong replay key")
        finally:
            self.submission_fence.release(submission)

    def _replay_prefill(  # noqa: PLR0912 - direct packed staging validates every row field
        self,
        key: TalkerPrefillCaptureKey,
        values: TalkerPrefillRowsExecutionInput | TalkerDecodeRowsExecutionInput,
    ) -> TalkerExecutionResult:
        if not isinstance(values, TalkerPrefillRowsExecutionInput):
            raise TypeError("Talker prefill requires a typed Talker prefill input")
        slot, lease = self._reserve(key, self._prefill, self._next_prefill)
        publications: tuple[PendingKVPublication, ...] = ()
        try:
            rows = len(values.request_ids)
            if len(values.rows) != rows:
                raise ValueError("Talker prefill row count does not match request IDs")
            lengths = tuple(row.text_token_ids.numel() for row in values.rows)
            if rows > key.capture_batch_size or sum(lengths) > key.token_capacity:
                raise ValueError("Talker prefill rows exceed the captured shape")
            if any(
                row.codec_token_ids.numel() != length
                or row.codec_token_mask.numel() != length
                for row, length in zip(values.rows, lengths, strict=True)
            ):
                raise ValueError("Talker prefill row tensors must have equal sequence lengths")
            self._validate_sampling_rows(tuple(row.sampling for row in values.rows))
            tokens = sum(lengths)
            self._stage_prefill_token_rows(slot, values.rows, tokens=tokens)
            slot.host_last.zero_()
            cursor = 0
            for index, length in enumerate(lengths):
                cursor += length
                slot.host_last[index] = cursor - 1
            slot.last.copy_(slot.host_last, non_blocking=True)
            slot.host_suppress.fill_(True)
            for index, row in enumerate(values.rows):
                slot.host_suppress[index] = row.suppress_eos
            slot.suppress.copy_(slot.host_suppress, non_blocking=True)
            self._stage_sampling_rows(
                slot.sampling,
                slot.host_sampling,
                tuple(row.sampling for row in values.rows),
            )
            source_offsets = slot.sampling.offsets[:rows]
            publications = self.cache.prepare_prefill(
                slot.context,
                key,
                values.request_ids,
                lengths,
            )
            return self._result(
                slot=slot,
                rows=rows,
                publications=publications,
                source_offsets=source_offsets,
            )
        except Exception:
            if publications:
                self.cache.abort(publications)
            raise
        finally:
            slot.lease_state.release(lease)

    def _replay_decode(  # noqa: PLR0912 - direct staging validates every row field
        self,
        key: TalkerDecodeCaptureKey,
        values: TalkerPrefillRowsExecutionInput | TalkerDecodeRowsExecutionInput,
    ) -> TalkerExecutionResult:
        if not isinstance(values, TalkerDecodeRowsExecutionInput):
            raise TypeError("Talker decode requires a typed Talker decode input")
        slot, lease = self._reserve(key, self._decode, self._next_decode)
        publications: tuple[PendingKVPublication, ...] = ()
        try:
            rows = len(values.request_ids)
            if len(values.rows) != rows:
                raise ValueError("Talker decode row count does not match request IDs")
            if rows > key.capture_batch_size:
                raise ValueError("Talker decode rows exceed the captured shape")
            if any(
                row.talker_step_embed.numel() != self.hidden_size
                for row in values.rows
            ):
                raise ValueError("talker_step_embed must match the captured hidden size")
            if any(row.talker_step_embed.device != self.device for row in values.rows):
                raise ValueError("Talker decode rows must be on the execution device")
            self._validate_sampling_rows(tuple(row.sampling for row in values.rows))
            self._stack_rows(
                slot.step_embed,
                tuple(row.talker_step_embed for row in values.rows),
            )
            host_text_rows = tuple(
                index
                for index, row in enumerate(values.rows)
                if row.text_token_id.device.type == "cpu"
            )
            device_text_rows = tuple(
                index
                for index, row in enumerate(values.rows)
                if row.text_token_id.device.type != "cpu"
            )
            if host_text_rows:
                slot.host_text.fill_(0)
                for index in host_text_rows:
                    slot.host_text[index] = values.rows[index].text_token_id.reshape(())
                if len(host_text_rows) == rows:
                    slot.text.copy_(slot.host_text, non_blocking=True)
                else:
                    slot.text.zero_()
                    for index in host_text_rows:
                        slot.text[index].copy_(slot.host_text[index], non_blocking=True)
            else:
                slot.text.zero_()
            if len(device_text_rows) == rows and rows > 1:
                torch.cat(
                    tuple(row.text_token_id.reshape(-1) for row in values.rows),
                    out=slot.text[:rows],
                )
            elif len(device_text_rows) == 1:
                index = device_text_rows[0]
                slot.text[index].copy_(values.rows[index].text_token_id.reshape(()))
            elif device_text_rows:
                for index in device_text_rows:
                    slot.text[index].copy_(values.rows[index].text_token_id.reshape(()))
            slot.host_suppress.fill_(True)
            for index, row in enumerate(values.rows):
                slot.host_suppress[index] = row.suppress_eos
            slot.suppress.copy_(slot.host_suppress, non_blocking=True)
            self._stage_sampling_rows(
                slot.sampling,
                slot.host_sampling,
                tuple(row.sampling for row in values.rows),
            )
            source_offsets = slot.sampling.offsets[:rows]
            publications = self.cache.prepare_decode(
                slot.context,
                key,
                values.request_ids,
                reuse_attention_plan=values.reuse_attention_plan,
            )
            return self._result(
                slot=slot,
                rows=rows,
                publications=publications,
                source_offsets=source_offsets,
            )
        except Exception:
            if publications:
                self.cache.abort(publications)
            raise
        finally:
            slot.lease_state.release(lease)

__all__ = ["TalkerExecutor", "TalkerKVBackend"]

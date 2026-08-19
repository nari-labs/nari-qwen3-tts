"""Typed static-buffer CUDA executor for the fixed Code Predictor frame."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from nari_qwen3_tts.contract.rng import (
    CodePredictorSamplerRoute,
    code_predictor_sampler_route,
)
from nari_qwen3_tts.contract.stage import CodePredictorCaptureKey
from nari_qwen3_tts.executor.cuda_graph import (
    CapturedCall,
    CaptureDriver,
    CudaGraphPoolFence,
    SlotLeaseState,
    TorchCaptureDriver,
)
from nari_qwen3_tts.executor.rows import CodePredictorExecutionRow, CodePredictorRowsExecutionInput
from nari_qwen3_tts.executor.types import CodePredictorInput, CodePredictorResult


@dataclass(slots=True)
class _CodePredictorSlot:
    values: CodePredictorInput
    fused_call: CapturedCall
    general_call: CapturedCall
    lease_state: SlotLeaseState
    host_temperature: torch.Tensor
    host_top_k: torch.Tensor
    host_top_p: torch.Tensor
    host_seed: torch.Tensor
    host_offsets: torch.Tensor
    default_position_ids: torch.Tensor
    positions_are_default: bool = True


class CodePredictorExecutor:
    """Capture the complete residual-code loop as one CUDA Graph per batch key."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        layer0_embedding: torch.nn.Embedding,
        config: object,
        max_batch_size: int,
        driver: CaptureDriver | None = None,
        submission_fence: CudaGraphPoolFence | None = None,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("Code Predictor max_batch_size must be a positive integer")
        self.model = model
        self.layer0_embedding = layer0_embedding
        self.config = config
        self.max_batch_size = max_batch_size
        self.hidden_size = int(config.talker.hidden_size)
        self.num_code_groups = int(config.num_code_groups)
        self.device = layer0_embedding.weight.device
        self.dtype = model.small_to_mtp_projection.weight.dtype
        self.driver = driver or TorchCaptureDriver(device=self.device, autocast_dtype=self.dtype)
        self.submission_fence = submission_fence or CudaGraphPoolFence(device=self.device)
        self._kv_cache: torch.Tensor | None = None
        self._projected_layer0: torch.Tensor | None = None
        self._projected_residual: torch.Tensor | None = None
        self._slots: dict[CodePredictorCaptureKey, _CodePredictorSlot] = {}

    @torch.no_grad()
    def _projected_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._projected_layer0 is None:
            projection = self.model.small_to_mtp_projection
            residual = self.model.model.codec_embedding
            expected = self.config.num_code_groups - 1
            if len(residual) != expected:
                raise ValueError("Code Predictor residual embedding count does not match config")
            self._projected_layer0 = projection(self.layer0_embedding.weight).contiguous()
            projected = [projection(residual[index].weight) for index in range(max(0, expected - 1))]
            self._projected_residual = (
                torch.stack(projected)
                if projected
                else projection.weight.new_empty((0, 0, 0))
            )
        assert self._projected_residual is not None
        return self._projected_layer0, self._projected_residual

    def _cache(self, rows: int, reference: torch.Tensor) -> torch.Tensor:
        if rows < 1:
            raise ValueError("Code Predictor jobs require at least one row")
        if rows > self.max_batch_size:
            raise ValueError(f"Code Predictor batch {rows} exceeds max_batch_size={self.max_batch_size}")
        cp = self.config.code_predictor
        expected = (
            cp.num_hidden_layers,
            self.max_batch_size,
            2,
            self.config.num_code_groups,
            cp.num_key_value_heads,
            cp.head_dim,
        )
        if (
            self._kv_cache is None
            or tuple(self._kv_cache.shape) != expected
            or self._kv_cache.device != reference.device
            or self._kv_cache.dtype != reference.dtype
        ):
            self._kv_cache = torch.zeros(expected, dtype=reference.dtype, device=reference.device)
        return self._kv_cache[:, :rows]

    @staticmethod
    def _sample_direct(
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        from nari_qwen3_tts.model.sampling import sample_logits_stateless

        return sample_logits_stateless(logits, temperature, top_k, top_p, seed, offset)

    @staticmethod
    def _require_sampling_domain(values: CodePredictorInput) -> None:
        for name, value in (("temperature", values.temperature), ("top_p", values.top_p)):
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"Code Predictor {name} must contain only finite values")
        if torch.any(values.temperature < 0):
            raise ValueError("Code Predictor temperature must be non-negative")
        if torch.any(values.top_k < 0):
            raise ValueError("Code Predictor top_k must be non-negative")
        if torch.any((values.top_p <= 0) | (values.top_p > 1)):
            raise ValueError("Code Predictor top_p must be in (0, 1]")
        if torch.any(values.seed < 0):
            raise ValueError("Code Predictor seed must be non-negative")
        if torch.any(values.offsets < 0):
            raise ValueError("Code Predictor offsets must be non-negative")

    def _capture_safe_sampling(
        self,
        values: CodePredictorInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enabled = values.temperature > 0
        vocab_size = self.config.code_predictor.vocab_size
        enabled_top_k = torch.where(
            values.top_k > 0,
            values.top_k.clamp(max=vocab_size),
            torch.full_like(values.top_k, vocab_size),
        )
        return (
            torch.where(enabled, values.temperature, torch.ones_like(values.temperature)),
            torch.where(enabled, enabled_top_k, torch.ones_like(values.top_k)),
            torch.where(enabled, values.top_p.clamp(max=1.0), torch.ones_like(values.top_p)),
        )

    @torch.inference_mode()
    def whole_frame(
        self,
        values: CodePredictorInput,
        *,
        capture_sampler: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
        | None = None,
    ) -> CodePredictorResult:
        configured_groups = int(self.config.num_code_groups)
        if values.num_code_groups != configured_groups:
            raise ValueError(
                "Code Predictor whole-frame jobs require the configured code groups: "
                f"got {values.num_code_groups}, expected {configured_groups}"
            )
        if capture_sampler is None:
            self._require_sampling_domain(values)
            sampler = self._sample_direct
            temperature, top_k, top_p = values.temperature, values.top_k, values.top_p
        else:
            sampler = capture_sampler
            temperature, top_k, top_p = self._capture_safe_sampling(values)
        rows = values.layer0_token.shape[0]
        dtype = self.model.small_to_mtp_projection.weight.dtype
        kv_cache = self._cache(rows, self.model.small_to_mtp_projection.weight)
        projected_layer0, projected_residual = self._projected_embeddings()
        position_ids = values.position_ids
        if position_ids is None:
            position_ids = (
                torch.arange(
                    values.num_code_groups,
                    dtype=torch.long,
                    device=values.layer0_token.device,
                )
                .unsqueeze(0)
                .expand(rows, -1)
            )
        frames = torch.empty(
            (rows, values.num_code_groups),
            dtype=torch.long,
            device=values.layer0_token.device,
        )
        frames[:, 0] = values.layer0_token
        self.model.forward_depth_unrolled(
            values.past_hidden.to(dtype).unsqueeze(1),
            position_ids[:, 0:1],
            kv_cache,
            cache_pos=0,
            apply_final_norm=False,
        )
        codec_sum = self.layer0_embedding(values.layer0_token).clone()
        projected_embed = F.embedding(values.layer0_token, projected_layer0)
        for group in range(1, values.num_code_groups):
            hidden = self.model.forward_depth_unrolled(
                projected_embed.unsqueeze(1),
                position_ids[:, group : group + 1],
                kv_cache,
                cache_pos=group,
                inputs_are_projected=True,
            ).squeeze(1)
            logits = torch.matmul(hidden, self.model.lm_head_weight[group - 1].t())
            token = sampler(
                logits,
                temperature,
                top_k,
                top_p,
                values.seed,
                values.offsets[:, group - 1],
            )
            frames[:, group] = token
            embed = self.model.model.codec_embedding[group - 1](token)
            codec_sum.add_(embed)
            if group < values.num_code_groups - 1:
                projected_embed = F.embedding(token, projected_residual[group - 1])
        return CodePredictorResult(frames=frames, codec_sum=codec_sum)

    @property
    def captured_cuda_graph_instances(self) -> int:
        return len(self._slots) * self.capture_instances_per_key

    @property
    def capture_instances_per_key(self) -> int:
        return 2

    def capture(self, key: CodePredictorCaptureKey) -> None:
        if not isinstance(key, CodePredictorCaptureKey):
            raise TypeError("Code Predictor executor received the wrong capture key")
        if key in self._slots:
            return
        rows = key.capture_batch_size
        values = CodePredictorInput(
            layer0_token=torch.zeros(rows, dtype=torch.long, device=self.device),
            past_hidden=torch.zeros((rows, self.hidden_size), dtype=self.dtype, device=self.device),
            temperature=torch.zeros(rows, device=self.device),
            top_k=torch.ones(rows, dtype=torch.int32, device=self.device),
            top_p=torch.ones(rows, device=self.device),
            seed=torch.zeros(rows, dtype=torch.long, device=self.device),
            offsets=torch.zeros(
                (rows, self.num_code_groups - 1),
                dtype=torch.long,
                device=self.device,
            ),
            num_code_groups=self.num_code_groups,
            position_ids=(
                torch.arange(self.num_code_groups, dtype=torch.long, device=self.device)
                .unsqueeze(0)
                .expand(rows, -1)
                .clone()
            ),
        )

        if self.device.type == "cuda":
            from nari_qwen3_tts.executor.sampling import (
                sample_code_predictor_cuda_graph as residual_sampler,
            )
            from nari_qwen3_tts.executor.sampling import (
                sample_logits_cuda_graph as general_sampler,
            )
        else:
            # The Triton kernel module cannot import off CUDA. The model's own
            # stateless sampler is its reference, and the CPU contract tests
            # capture through it.
            from nari_qwen3_tts.model.sampling import (
                sample_logits_stateless as residual_sampler,
            )
            general_sampler = residual_sampler

        def operation(sampler) -> tuple[torch.Tensor, torch.Tensor]:
            result = self.whole_frame(values, capture_sampler=sampler)
            return result.frames, result.codec_sum

        self._slots[key] = _CodePredictorSlot(
            values,
            self.driver.capture(lambda: operation(residual_sampler)),
            self.driver.capture(lambda: operation(general_sampler)),
            SlotLeaseState(),
            torch.empty(rows, device="cpu", pin_memory=self.device.type == "cuda"),
            torch.empty(rows, dtype=torch.int32, device="cpu", pin_memory=self.device.type == "cuda"),
            torch.empty(rows, device="cpu", pin_memory=self.device.type == "cuda"),
            torch.empty(rows, dtype=torch.long, device="cpu", pin_memory=self.device.type == "cuda"),
            torch.empty(
                (rows, self.num_code_groups - 1),
                dtype=torch.long,
                device="cpu",
                pin_memory=self.device.type == "cuda",
            ),
            values.position_ids.clone(),
        )

    def replay(
        self,
        key: CodePredictorCaptureKey,
        values: CodePredictorRowsExecutionInput,
    ) -> CodePredictorResult:
        submission = self.submission_fence.reserve()
        try:
            return self._replay_owned(key, values)
        finally:
            self.submission_fence.release(submission)

    @staticmethod
    def _pack_host_vector(
        destination: torch.Tensor,
        rows: tuple[CodePredictorExecutionRow, ...],
        *,
        field: str,
        padding: float | int,
    ) -> None:
        """Pack Python row metadata without one Torch dispatch per scalar."""

        view = destination.numpy()
        logical_rows = len(rows)
        view[:logical_rows] = tuple(getattr(row, field) for row in rows)
        view[logical_rows:] = padding

    @staticmethod
    def _pack_host_offsets(
        destination: torch.Tensor,
        rows: tuple[CodePredictorExecutionRow, ...],
    ) -> None:
        """Pack the fixed fifteen-offset rows through one contiguous host write."""

        view = destination.numpy()
        logical_rows = len(rows)
        view[:logical_rows] = tuple(row.offsets for row in rows)
        view[logical_rows:] = 0

    def _replay_owned(  # noqa: PLR0912,PLR0915 - fixed validation and staging
        self,
        key: CodePredictorCaptureKey,
        values: CodePredictorRowsExecutionInput,
    ) -> CodePredictorResult:
        if not isinstance(key, CodePredictorCaptureKey):
            raise TypeError("Code Predictor executor received the wrong capture key")
        if not isinstance(values, CodePredictorRowsExecutionInput):
            raise TypeError("Code Predictor replay requires a typed Code Predictor input")
        slot = self._slots.get(key)
        if slot is None:
            raise RuntimeError("Code Predictor CUDA Graph has not been captured")
        logical_rows = len(values.rows)
        if logical_rows > key.capture_batch_size:
            raise ValueError("Code Predictor input exceeds its captured shape")
        lease = slot.lease_state.reserve()
        try:
            if any(len(row.offsets) != self.num_code_groups - 1 for row in values.rows):
                raise ValueError("Code Predictor row RNG offsets do not cover every residual codebook")
            row_routes = {
                code_predictor_sampler_route(
                    temperature=row.temperature,
                    top_k=row.top_k,
                )
                for row in values.rows
            }
            if row_routes != {values.sampler_route}:
                raise ValueError(
                    "Code Predictor rows must match one homogeneous sampler route"
                )
            if any(row.past_hidden.numel() != self.hidden_size for row in values.rows):
                raise ValueError("Code Predictor row past_hidden does not match the captured hidden size")
            if any(row.layer0_token.device != self.device for row in values.rows):
                raise ValueError("Code Predictor rows must be on the execution device")
            if any(
                row.position_ids is not None
                and (
                    row.position_ids.numel() != self.num_code_groups
                    or row.position_ids.dtype is not torch.long
                )
                for row in values.rows
            ):
                raise ValueError("Code Predictor row position IDs do not match the captured shape")
            slot.values.layer0_token.zero_()
            slot.values.past_hidden.zero_()
            if len(values.rows) == 1:
                slot.values.layer0_token[0].copy_(values.rows[0].layer0_token.reshape(()))
                slot.values.past_hidden[0].copy_(values.rows[0].past_hidden.reshape(-1))
            elif values.rows:
                torch.stack(
                    tuple(row.layer0_token.reshape(()) for row in values.rows),
                    out=slot.values.layer0_token[:logical_rows],
                )
                torch.stack(
                    tuple(row.past_hidden.reshape(-1) for row in values.rows),
                    out=slot.values.past_hidden[:logical_rows],
                )
            assert slot.values.position_ids is not None
            slot.values.position_ids.copy_(slot.default_position_ids)
            for row_index, row in enumerate(values.rows):
                if row.position_ids is not None:
                    slot.values.position_ids[row_index].copy_(row.position_ids.reshape(-1))
            slot.positions_are_default = all(row.position_ids is None for row in values.rows)
            scalar_fields = (
                (slot.host_temperature, slot.values.temperature, 0.0, "temperature"),
                (slot.host_top_k, slot.values.top_k, 1, "top_k"),
                (slot.host_top_p, slot.values.top_p, 1.0, "top_p"),
                (slot.host_seed, slot.values.seed, 0, "seed"),
            )
            for host, device, padding, name in scalar_fields:
                self._pack_host_vector(
                    host,
                    values.rows,
                    field=name,
                    padding=padding,
                )
                device.copy_(host, non_blocking=True)
            self._pack_host_offsets(slot.host_offsets, values.rows)
            slot.values.offsets.copy_(slot.host_offsets, non_blocking=True)
            captured_call = (
                slot.general_call
                if values.sampler_route is CodePredictorSamplerRoute.GENERAL
                else slot.fused_call
            )
            captured = captured_call.replay()
            if not isinstance(captured, tuple) or len(captured) != 2:
                raise RuntimeError("Code Predictor CUDA Graph returned an invalid result")
            frames, codec_sum = captured
            return CodePredictorResult(
                frames=frames[:logical_rows].clone(),
                codec_sum=codec_sum[:logical_rows].clone(),
            )
        finally:
            slot.lease_state.release(lease)


__all__ = ["CodePredictorExecutor"]

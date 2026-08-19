"""Single-GPU transformer primitives with checkpoint-compatible names."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def quantize_fp8_weight_blocks(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a row-major gate/up matrix into DeepGEMM 128x128 blocks."""

    if weight.ndim != 2 or not weight.is_contiguous():
        raise ValueError("FP8 block quantization requires a contiguous matrix")
    rows, columns = weight.shape
    if rows % 128 or columns % 128:
        raise ValueError(
            f"FP8 block quantization requires dimensions divisible by 128, got {tuple(weight.shape)}"
        )
    blocks = weight.view(rows // 128, 128, columns // 128, 128)
    maximum = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp_(1e-4)
    scales = torch.pow(2.0, torch.ceil(torch.log2(maximum / 448.0)))
    quantized = (blocks / scales).to(torch.float8_e4m3fn)
    return quantized.view_as(weight).contiguous(), scales.view(rows // 128, columns // 128)


def qwen3_tts_silu_and_mul(
    gate_up: torch.Tensor,
    *,
    use_cuda_kernel: bool = True,
) -> torch.Tensor:
    """Fuse only the pointwise SwiGLU operation on supported CUDA inputs."""

    if use_cuda_kernel and gate_up.is_cuda and gate_up.dtype in (torch.float16, torch.bfloat16):
        from flashinfer.activation import silu_and_mul

        return silu_and_mul(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def qwen3_tts_add_rmsnorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: "RMSNorm",
    *,
    use_cuda_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the attention residual and apply the following RMS normalization.

    FlashInfer updates both CUDA inputs in place: ``residual`` becomes the
    residual sum and ``hidden_states`` becomes its normalized view. The guard
    below is the only dispatch -- callers write one loop, and unsupported
    devices, dtypes, or layouts fall through to the reference math here rather
    than to a separately maintained copy of the caller.
    """

    if (
        use_cuda_kernel
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and hidden_states.is_contiguous()
        and residual.is_contiguous()
        and hidden_states.shape == residual.shape
        and hidden_states.data_ptr() != residual.data_ptr()
    ):
        from flashinfer.norm import fused_add_rmsnorm

        hidden_size = hidden_states.shape[-1]
        fused_add_rmsnorm(
            hidden_states.view(-1, hidden_size),
            residual.view(-1, hidden_size),
            norm.weight,
            eps=norm.variance_epsilon,
        )
        return hidden_states, residual
    residual = residual + hidden_states
    return norm(residual), residual


def qwen3_tts_rmsnorm(
    hidden_states: torch.Tensor,
    norm: "RMSNorm",
    *,
    use_cuda_kernel: bool = True,
) -> torch.Tensor:
    """Apply one RMS normalization through the fused CUDA boundary."""

    if use_cuda_kernel and hidden_states.is_cuda and hidden_states.dtype in (torch.float16, torch.bfloat16):
        from flashinfer.norm import rmsnorm

        shape = hidden_states.shape
        normalized = rmsnorm(
            hidden_states.reshape(-1, shape[-1]),
            norm.weight,
            eps=norm.variance_epsilon,
        )
        return normalized.reshape(shape)
    return norm(hidden_states)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized * self.weight.float()).to(original_dtype)


class PackedLinear(nn.Module):
    """One fixed packed parameter populated from named checkpoint shards."""

    def __init__(self, input_size: int, output_sizes: tuple[int, ...], *, bias: bool = False) -> None:
        super().__init__()
        if not output_sizes or any(size < 1 for size in output_sizes):
            raise ValueError("output_sizes must contain positive values")
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.weight = nn.Parameter(torch.empty(sum(output_sizes), input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(sum(output_sizes)))
        else:
            self.register_parameter("bias", None)
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: str | int | None = None,
    ) -> None:
        if not isinstance(shard_id, int) or not 0 <= shard_id < len(self.output_sizes):
            raise ValueError(f"packed linear requires an integer shard id, got {shard_id!r}")
        offset = sum(self.output_sizes[:shard_id])
        destination = parameter.data.narrow(0, offset, self.output_sizes[shard_id])
        if destination.shape != loaded.shape:
            raise ValueError(f"packed shard shape mismatch: {tuple(destination.shape)} != {tuple(loaded.shape)}")
        destination.copy_(loaded)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class PackedQKVLinear(PackedLinear):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        super().__init__(
            hidden_size,
            (
                num_heads * head_dim,
                num_kv_heads * head_dim,
                num_kv_heads * head_dim,
            ),
            bias=False,
        )

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: str | int | None = None,
    ) -> None:
        mapping = {"q": 0, "k": 1, "v": 2}
        if shard_id not in mapping:
            raise ValueError(f"QKV shard id must be q, k, or v; got {shard_id!r}")
        super().weight_loader(parameter, loaded, mapping[shard_id])


class GatedMLP(nn.Module):
    """Qwen SwiGLU MLP that can serve its gate/up projection from FP8 blocks.

    FP8 weights are prepared once after loading through
    :meth:`initialize_fp8_gate_up_weight`. Both weight formats run through the
    same ``forward`` method.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = PackedLinear(hidden_size, (intermediate_size, intermediate_size), bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.register_buffer("_fp8_gate_up_weight", None, persistent=False)
        self.register_buffer("_fp8_gate_up_scale", None, persistent=False)
        self._cuda_optimized_math_enabled = False

    @property
    def fp8_enabled(self) -> bool:
        return self._fp8_gate_up_weight is not None and self._fp8_gate_up_scale is not None

    @property
    def cuda_optimized_math_enabled(self) -> bool:
        return self._cuda_optimized_math_enabled

    def enable_cuda_optimized_math(self) -> None:
        """Enable model-owned CUDA kernels after execution preparation."""

        self._cuda_optimized_math_enabled = True

    @torch.no_grad()
    def initialize_fp8_gate_up_weight(self) -> bool:
        """Quantize gate/up into DeepGEMM blocks when the device supports it."""

        weight = self.gate_up_proj.weight
        if (
            weight.device.type != "cuda"
            or weight.dtype != torch.bfloat16
            or torch.cuda.get_device_capability(weight.device)[0] != 9
        ):
            return False
        quantized, scales = quantize_fp8_weight_blocks(weight)
        self._fp8_gate_up_weight = quantized
        self._fp8_gate_up_scale = scales
        return True

    def _project_gate_up(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized = self._fp8_gate_up_weight
        scales = self._fp8_gate_up_scale
        if (
            quantized is None
            or scales is None
            or inputs.device.type != "cuda"
            or inputs.device != quantized.device
            or inputs.dtype != torch.bfloat16
            or not inputs.is_contiguous()
        ):
            return self.gate_up_proj(inputs)
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        flat = inputs.view(-1, inputs.shape[-1])
        projected = fp8_blockscale_gemm_sm90(
            flat,
            quantized,
            weight_scale=scales,
            out_dtype=torch.bfloat16,
        )
        return projected.view(*inputs.shape[:-1], quantized.shape[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            qwen3_tts_silu_and_mul(
                self._project_gate_up(inputs),
                use_cuda_kernel=self.cuda_optimized_math_enabled,
            )
        )


class Attention(nn.Module):
    """Parameter-owning attention with an executor-supplied context."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.qkv_proj = PackedQKVLinear(hidden_size, num_heads, num_kv_heads, head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_count = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        return (
            query.view(token_count, self.num_heads, self.head_dim),
            key.view(token_count, self.num_kv_heads, self.head_dim),
            value.view(token_count, self.num_kv_heads, self.head_dim),
        )

    def forward(self, hidden_states: torch.Tensor, attention_context: object) -> torch.Tensor:
        """Project QKV and delegate the remaining attention math to the context.

        ``attend`` receives this module so the context can apply ``q_norm``,
        ``k_norm``, and rotary embedding (``rope_theta``) to the raw
        projections before attending; this forward never applies them itself.
        """
        query, key, value = self.project_qkv(hidden_states)
        output = attention_context.attend(self, query, key, value)
        return self.o_proj(output.reshape(hidden_states.shape[0], -1))


__all__ = ["Attention", "GatedMLP", "PackedLinear", "PackedQKVLinear", "RMSNorm"]

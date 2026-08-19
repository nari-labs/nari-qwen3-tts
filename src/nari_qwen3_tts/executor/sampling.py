"""CUDA Graph-only sampling after typed executor validation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from nari_qwen3_tts.contract.rng import MAX_FUSED_RESIDUAL_TOP_K

MAX_RESIDUAL_TOP_K = MAX_FUSED_RESIDUAL_TOP_K

# The kernel's bucket width is the request domain's admitted top-k ceiling, so
# every accepted request reaches the one fused residual sampler.
MAX_CODE_PREDICTOR_TOP_K = MAX_RESIDUAL_TOP_K


@triton.autotune(
    configs=[
        triton.Config({"block_size": 4096}, num_warps=4, num_stages=2),
        triton.Config({"block_size": 8192}, num_warps=4, num_stages=2),
        triton.Config({"block_size": 8192}, num_warps=8, num_stages=2),
        triton.Config({"block_size": 16384}, num_warps=8, num_stages=2),
        triton.Config({"block_size": 16384}, num_warps=16, num_stages=2),
        triton.Config({"block_size": 32768}, num_warps=16, num_stages=2),
        triton.Config({"block_size": 32768}, num_warps=32, num_stages=2),
    ],
    key=["vocab_size", "apply_penalty"],
)
@triton.jit
def _fused_talker_sampling_prep_kernel(
    logits_ptr,
    temperature_ptr,
    penalty_ptr,
    seen_mask_ptr,
    probabilities_ptr,
    vocab_size,
    logits_stride_b,
    logits_stride_v,
    probabilities_stride_b,
    probabilities_stride_v,
    seen_stride_b,
    seen_stride_v,
    block_size: tl.constexpr,
    apply_penalty: tl.constexpr,
):
    row = tl.program_id(0)
    inverse_temperature = 1.0 / tl.load(temperature_ptr + row)
    if apply_penalty:
        penalty = tl.load(penalty_ptr + row)

    maximum = -float("inf")
    for start in range(0, vocab_size, block_size):
        offsets = start + tl.arange(0, block_size)
        valid = offsets < vocab_size
        values = tl.load(
            logits_ptr + row * logits_stride_b + offsets * logits_stride_v,
            mask=valid,
            other=-float("inf"),
        )
        if apply_penalty:
            seen = tl.load(
                seen_mask_ptr + row * seen_stride_b + offsets * seen_stride_v,
                mask=valid,
                other=0,
            ).to(tl.int1)
            penalized = tl.where(values > 0, values / penalty, values * penalty)
            values = tl.where(seen, penalized, values)
        maximum = tl.maximum(maximum, tl.max(tl.where(valid, values, -float("inf"))))

    scaled_maximum = maximum * inverse_temperature
    exponential_sum = tl.zeros([], dtype=tl.float32)
    for start in range(0, vocab_size, block_size):
        offsets = start + tl.arange(0, block_size)
        valid = offsets < vocab_size
        values = tl.load(
            logits_ptr + row * logits_stride_b + offsets * logits_stride_v,
            mask=valid,
            other=0.0,
        )
        if apply_penalty:
            seen = tl.load(
                seen_mask_ptr + row * seen_stride_b + offsets * seen_stride_v,
                mask=valid,
                other=0,
            ).to(tl.int1)
            penalized = tl.where(values > 0, values / penalty, values * penalty)
            values = tl.where(seen, penalized, values)
        exponentials = tl.exp(values * inverse_temperature - scaled_maximum)
        exponentials = tl.where(valid, exponentials, 0.0)
        exponential_sum += tl.sum(exponentials)

    inverse_sum = 1.0 / tl.maximum(exponential_sum, 1e-30)
    for start in range(0, vocab_size, block_size):
        offsets = start + tl.arange(0, block_size)
        valid = offsets < vocab_size
        values = tl.load(
            logits_ptr + row * logits_stride_b + offsets * logits_stride_v,
            mask=valid,
            other=0.0,
        )
        if apply_penalty:
            seen = tl.load(
                seen_mask_ptr + row * seen_stride_b + offsets * seen_stride_v,
                mask=valid,
                other=0,
            ).to(tl.int1)
            penalized = tl.where(values > 0, values / penalty, values * penalty)
            values = tl.where(seen, penalized, values)
        probabilities = tl.exp(values * inverse_temperature - scaled_maximum) * inverse_sum
        tl.store(
            probabilities_ptr + row * probabilities_stride_b + offsets * probabilities_stride_v,
            probabilities,
            mask=valid,
        )


def _talker_probabilities(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    repetition_penalty: torch.Tensor,
    seen_token_mask: torch.Tensor,
) -> torch.Tensor:
    rows, vocab_size = logits.shape
    probabilities = torch.empty_like(logits, dtype=torch.float32)
    with torch.cuda.device(logits.device):
        # ``block_size`` comes from @triton.autotune, not from this call. The
        # first launch for a given key benchmarks every config (do_bench),
        # which can leave ``probabilities`` unordered on the current stream
        # relative to the downstream FlashInfer read -> garbage. Gating the
        # sync on autotune alone is not enough; pairing it with the device
        # context above is what fixes it. Detect the autotune call by the
        # config cache growing, so steady state stays sync-free.
        cache = getattr(_fused_talker_sampling_prep_kernel, "cache", None)
        cache_size = len(cache) if cache is not None else 0
        _fused_talker_sampling_prep_kernel[(rows,)](
            logits,
            temperature,
            repetition_penalty,
            seen_token_mask,
            probabilities,
            vocab_size,
            logits.stride(0),
            logits.stride(1),
            probabilities.stride(0),
            probabilities.stride(1),
            seen_token_mask.stride(0),
            seen_token_mask.stride(1),
            apply_penalty=True,
        )
        if cache is not None and len(cache) > cache_size:
            torch.cuda.current_stream().synchronize()
    return probabilities


def sample_talker_cuda_graph(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    repetition_penalty: torch.Tensor,
    seen_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Sample each logical request through physical row zero inside one CUDA Graph."""

    if not logits.is_cuda:
        return logits.argmax(dim=-1).to(torch.long)
    greedy = temperature == 0
    safe_temperature = torch.where(greedy, torch.ones_like(temperature), temperature)
    enabled_k = torch.where(
        top_k > 0,
        top_k.clamp(max=logits.shape[1]),
        torch.full_like(top_k, logits.shape[1]),
    )
    safe_k = torch.where(greedy, torch.ones_like(top_k), enabled_k)
    safe_p = torch.where(greedy, torch.ones_like(top_p), top_p)
    import flashinfer

    return torch.cat(
        [
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                _talker_probabilities(
                    logits[row : row + 1],
                    safe_temperature[row : row + 1],
                    repetition_penalty[row : row + 1],
                    seen_token_mask[row : row + 1],
                ),
                safe_k[row : row + 1],
                safe_p[row : row + 1],
                deterministic=True,
                seed=seed[row : row + 1],
                offset=offset[row : row + 1],
            )
            for row in range(logits.shape[0])
        ]
    ).to(torch.long)


def sample_logits_cuda_graph(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Run a fixed row-zero FlashInfer draw for every captured physical row."""

    greedy = temperature == 0
    safe_temperature = torch.where(greedy, torch.ones_like(temperature), temperature)
    enabled_k = torch.where(
        top_k > 0,
        top_k.clamp(max=logits.shape[1]),
        torch.full_like(top_k, logits.shape[1]),
    )
    safe_k = torch.where(greedy, torch.ones_like(top_k), enabled_k)
    safe_p = torch.where(greedy, torch.ones_like(top_p), top_p)
    scaled = logits / safe_temperature.unsqueeze(1)
    if logits.is_cuda:
        import flashinfer

        sampled = torch.cat(
            [
                flashinfer.sampling.top_k_top_p_sampling_from_logits(
                    scaled[row : row + 1],
                    safe_k[row : row + 1],
                    safe_p[row : row + 1],
                    deterministic=True,
                    seed=seed[row : row + 1],
                    offset=offset[row : row + 1],
                )
                for row in range(logits.shape[0])
            ]
        ).to(torch.long)
    else:
        sampled = scaled.argmax(dim=-1).to(torch.long)
    return torch.where(greedy, logits.argmax(dim=-1), sampled)


@triton.jit
def _sample_code_predictor_top_k_top_p_kernel(
    logits_ptr,
    temperature_ptr,
    top_k_ptr,
    top_p_ptr,
    seed_ptr,
    offset_ptr,
    output_ptr,
    logits_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    block_size: tl.constexpr,
    max_top_k: tl.constexpr,
):
    row = tl.program_id(0)
    token_ids = tl.arange(0, block_size)
    valid_token = token_ids < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + token_ids,
        mask=valid_token,
        other=-float("inf"),
    ).to(tl.float32)
    temperature = tl.load(temperature_ptr + row).to(tl.float32)
    sampling_enabled = temperature > 0.0
    scaled = logits / tl.where(sampling_enabled, temperature, 1.0)
    top_values = tl.topk(scaled, max_top_k)
    ranks = tl.arange(0, max_top_k)
    requested_top_k = tl.load(top_k_ptr + row)
    active_rank = ranks < tl.where(sampling_enabled, requested_top_k, 1)
    weights = tl.exp(top_values - tl.max(top_values))
    weights = tl.where(active_rank, weights, 0.0)
    total = tl.sum(weights)
    cumulative = tl.cumsum(weights)
    top_p = tl.minimum(tl.load(top_p_ptr + row).to(tl.float32), 1.0)
    nucleus_rank = active_rank & ((cumulative - weights) < top_p * total)
    nucleus_mass = tl.sum(tl.where(nucleus_rank, weights, 0.0))
    seed_value = tl.load(seed_ptr + row)
    offset_value = tl.load(offset_ptr + row * offset_stride)
    uniform = tl.rand(seed_value, offset_value)
    selected_rank = tl.min(
        tl.where(
            nucleus_rank & (cumulative >= uniform * nucleus_mass),
            ranks,
            max_top_k,
        )
    )
    selected_value = tl.sum(tl.where(ranks == selected_rank, top_values, 0.0))
    selected_tie_ordinal = tl.sum(
        tl.where((ranks < selected_rank) & (top_values == selected_value), 1, 0)
    )
    value_matches = valid_token & (scaled == selected_value)
    token_tie_ordinal = tl.cumsum(value_matches.to(tl.int32)) - 1
    selected_token = tl.min(
        tl.where(
            value_matches & (token_tie_ordinal == selected_tie_ordinal),
            token_ids,
            block_size,
        )
    )
    tl.store(output_ptr + row, selected_token)


def sample_code_predictor_cuda_graph(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Run whole-frame sampling with one logical RNG row per request."""

    if not logits.is_cuda:
        return logits.argmax(dim=-1).to(torch.long)
    rows, vocab_size = logits.shape
    output = torch.empty(rows, dtype=torch.long, device=logits.device)
    block_size = triton.next_power_of_2(vocab_size)
    with torch.cuda.device(logits.device):
        _sample_code_predictor_top_k_top_p_kernel[(rows,)](
            logits,
            temperature,
            top_k,
            top_p,
            seed,
            offset,
            output,
            logits.stride(0),
            offset.stride(0),
            vocab_size=vocab_size,
            block_size=block_size,
            max_top_k=MAX_CODE_PREDICTOR_TOP_K,
            num_warps=8,
        )
    return output


__all__ = [
    "MAX_CODE_PREDICTOR_TOP_K",
    "sample_code_predictor_cuda_graph",
    "sample_logits_cuda_graph",
    "sample_talker_cuda_graph",
]

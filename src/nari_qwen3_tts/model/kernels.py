"""Triton kernels for Qwen3-TTS attention math.

These model-local kernels implement QK normalization, rotary embedding, and
short-cache attention. Model forward methods dispatch to them directly.
Importing this module requires Triton, so model code imports it lazily inside
its CUDA branch and stays importable on CPU-only hosts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paired_qk_norm_rope_paged_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    q_weight_ptr,
    k_weight_ptr,
    position_ptr,
    write_page_ptr,
    write_offset_ptr,
    eps,
    rope_theta,
    sq_m,
    sq_h,
    sq_d,
    sk_m,
    sk_h,
    sk_d,
    sv_m,
    sv_h,
    sv_d,
    sc_page,
    sc_kind,
    sc_token,
    sc_head,
    sc_d,
    h_q: tl.constexpr,
    h_k: tl.constexpr,
    d: tl.constexpr,
    half_d: tl.constexpr,
):
    token = tl.program_id(0)
    paired_head = tl.program_id(1)
    is_query = paired_head < h_q
    head = tl.where(is_query, paired_head, paired_head - h_q)
    base = tl.where(
        is_query,
        q_ptr + token * sq_m + head * sq_h,
        k_ptr + token * sk_m + head * sk_h,
    )
    weight = tl.where(is_query, q_weight_ptr, k_weight_ptr)
    stride_d = tl.where(is_query, sq_d, sk_d)
    offsets = tl.arange(0, half_d)
    first = tl.load(base + offsets * stride_d).to(tl.float32)
    second = tl.load(base + (offsets + half_d) * stride_d).to(tl.float32)
    sum_squares = tl.sum(first * first, axis=0) + tl.sum(second * second, axis=0)
    inverse_rms = 1.0 / tl.sqrt(sum_squares / d + eps)
    first = first * inverse_rms * tl.load(weight + offsets).to(tl.float32)
    second = second * inverse_rms * tl.load(weight + offsets + half_d).to(tl.float32)
    element_type = q_ptr.dtype.element_ty
    first = first.to(element_type).to(tl.float32)
    second = second.to(element_type).to(tl.float32)
    position = tl.load(position_ptr + token).to(tl.float32)
    inverse_frequency = tl.exp(-tl.log(rope_theta) * (offsets.to(tl.float32) * 2.0 / d))
    angle = position * inverse_frequency
    cosine = tl.cos(angle)
    sine = tl.sin(angle)
    out_first = first * cosine - second * sine
    out_second = second * cosine + first * sine
    tl.store(base + offsets * stride_d, out_first.to(element_type))
    tl.store(base + (offsets + half_d) * stride_d, out_second.to(element_type))

    page = tl.load(write_page_ptr + token)
    page_offset = tl.load(write_offset_ptr + token)
    is_key = ~is_query
    key_cache = cache_ptr + page * sc_page + page_offset * sc_token + head * sc_head
    tl.store(key_cache + offsets * sc_d, out_first.to(element_type), mask=is_key)
    tl.store(key_cache + (offsets + half_d) * sc_d, out_second.to(element_type), mask=is_key)

    copies_value = is_query & (paired_head < h_k)
    value_base = v_ptr + token * sv_m + paired_head * sv_h
    value_cache = cache_ptr + page * sc_page + sc_kind + page_offset * sc_token + paired_head * sc_head
    value_first = tl.load(value_base + offsets * sv_d, mask=copies_value, other=0.0)
    value_second = tl.load(value_base + (offsets + half_d) * sv_d, mask=copies_value, other=0.0)
    tl.store(value_cache + offsets * sc_d, value_first, mask=copies_value)
    tl.store(value_cache + (offsets + half_d) * sc_d, value_second, mask=copies_value)


def paired_qk_norm_rope_paged_cache_(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    position_ids: torch.Tensor,
    epsilon: float,
    rope_theta: float,
    layer_cache: torch.Tensor,
    write_pages: torch.Tensor,
    write_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize/rotate QK in place and stage K/V into paged cache once."""

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Qwen3-TTS QKV tensors must have rank three")
    tokens, query_heads, head_dim = query.shape
    if key.shape != value.shape or key.shape[0] != tokens or key.shape[2] != head_dim:
        raise ValueError("Qwen3-TTS QKV shapes are incompatible")
    key_heads = key.shape[1]
    if query.device.type != "cuda" or not (query.device == key.device == value.device == layer_cache.device):
        raise ValueError("paired QKV paged staging requires one CUDA device")
    if not (query.dtype == key.dtype == value.dtype == layer_cache.dtype):
        raise TypeError("paired QKV paged staging requires one dtype")
    if head_dim % 2 or query_weight.shape != (head_dim,) or key_weight.shape != (head_dim,):
        raise ValueError("QK norm weights must match an even head dimension")
    if position_ids.shape != (tokens,) or write_pages.shape != (tokens,) or write_offsets.shape != (tokens,):
        raise ValueError("paged QKV metadata must contain one entry per token")
    if layer_cache.ndim != 5 or layer_cache.shape[1] != 2 or layer_cache.shape[3:] != (key_heads, head_dim):
        raise ValueError("paged QKV cache has an invalid layout")
    if not all(tensor.stride(-1) == 1 for tensor in (query, key, value, layer_cache)):
        raise ValueError("QKV and paged cache must be contiguous per head")
    _paired_qk_norm_rope_paged_cache_kernel[(tokens, query_heads + key_heads)](
        query,
        key,
        value,
        layer_cache,
        query_weight,
        key_weight,
        position_ids,
        write_pages,
        write_offsets,
        epsilon,
        rope_theta,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        layer_cache.stride(0),
        layer_cache.stride(1),
        layer_cache.stride(2),
        layer_cache.stride(3),
        layer_cache.stride(4),
        h_q=query_heads,
        h_k=key_heads,
        d=head_dim,
        half_d=head_dim // 2,
        num_warps=4,
        num_stages=1,
    )
    return query, key


@triton.jit
def _paired_qk_norm_rope_dense_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    q_weight_ptr,
    k_weight_ptr,
    position_ptr,
    epsilon,
    rope_theta,
    cache_position,
    sp_m,
    sq_m,
    sq_h,
    sq_d,
    sk_m,
    sk_h,
    sk_d,
    sv_m,
    sv_h,
    sv_d,
    skc_b,
    skc_s,
    skc_h,
    skc_d,
    svc_b,
    svc_s,
    svc_h,
    svc_d,
    h_q: tl.constexpr,
    h_k: tl.constexpr,
    d: tl.constexpr,
    half_d: tl.constexpr,
    round_to_input: tl.constexpr,
    write_cache: tl.constexpr,
):
    token = tl.program_id(0)
    paired_head = tl.program_id(1)
    is_query = paired_head < h_q
    head = tl.where(is_query, paired_head, paired_head - h_q)
    base = tl.where(
        is_query,
        q_ptr + token * sq_m + head * sq_h,
        k_ptr + token * sk_m + head * sk_h,
    )
    weight = tl.where(is_query, q_weight_ptr, k_weight_ptr)
    stride_d = tl.where(is_query, sq_d, sk_d)
    offsets = tl.arange(0, half_d)
    first = tl.load(base + offsets * stride_d).to(tl.float32)
    second = tl.load(base + (offsets + half_d) * stride_d).to(tl.float32)
    sum_squares = tl.sum(first * first, axis=0) + tl.sum(
        second * second,
        axis=0,
    )
    inverse_rms = 1.0 / tl.sqrt(sum_squares / d + epsilon)
    weight_first = tl.load(weight + offsets).to(tl.float32)
    weight_second = tl.load(weight + offsets + half_d).to(tl.float32)
    element_type = q_ptr.dtype.element_ty
    first = first * inverse_rms * weight_first
    second = second * inverse_rms * weight_second
    if round_to_input:
        first = first.to(element_type).to(tl.float32)
        second = second.to(element_type).to(tl.float32)
    position = tl.load(position_ptr + token * sp_m).to(tl.float32)
    inverse_frequency = tl.exp(-tl.log(rope_theta) * (offsets.to(tl.float32) * 2.0 / d))
    angle = position * inverse_frequency
    cosine = tl.cos(angle)
    sine = tl.sin(angle)
    out_first = first * cosine - second * sine
    out_second = second * cosine + first * sine
    tl.store(base + offsets * stride_d, out_first.to(element_type))
    tl.store(base + (offsets + half_d) * stride_d, out_second.to(element_type))
    if write_cache:
        is_key = ~is_query
        key_cache = k_cache_ptr + token * skc_b + cache_position * skc_s + head * skc_h
        tl.store(key_cache + offsets * skc_d, out_first.to(element_type), mask=is_key)
        tl.store(key_cache + (offsets + half_d) * skc_d, out_second.to(element_type), mask=is_key)
        copies_value = is_query & (paired_head < h_k)
        value_base = v_ptr + token * sv_m + paired_head * sv_h
        value_cache = v_cache_ptr + token * svc_b + cache_position * svc_s + paired_head * svc_h
        value_first = tl.load(value_base + offsets * sv_d, mask=copies_value, other=0.0)
        value_second = tl.load(value_base + (offsets + half_d) * sv_d, mask=copies_value, other=0.0)
        tl.store(value_cache + offsets * svc_d, value_first, mask=copies_value)
        tl.store(value_cache + (offsets + half_d) * svc_d, value_second, mask=copies_value)


def _validate_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[int, int, int, int]:
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("QK tensors must have shape [tokens, heads, head_dim]")
    tokens, query_heads, head_dim = query.shape
    if key.shape[0] != tokens or key.shape[2] != head_dim or head_dim % 2:
        raise ValueError("QK shapes are incompatible")
    key_heads = key.shape[1]
    if query.device.type != "cuda" or key.device != query.device or query.dtype != key.dtype:
        raise ValueError("paired QK transform requires one CUDA dtype/device")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("QK tensors must be contiguous per head")
    if positions.shape != (tokens,) or positions.device != query.device:
        raise ValueError("positions must contain one CUDA value per token")
    if query_weight.shape != (head_dim,) or key_weight.shape != (head_dim,):
        raise ValueError("QK norm weights must match head_dim")
    return tokens, query_heads, key_heads, head_dim


def paired_qk_norm_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    epsilon: float,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the paired QK transform with its required BF16 rounding boundary."""

    tokens, query_heads, key_heads, head_dim = _validate_qk(
        query,
        key,
        query_weight,
        key_weight,
        positions,
    )
    _paired_qk_norm_rope_dense_kernel[(tokens, query_heads + key_heads)](
        query,
        key,
        query,
        query,
        query,
        query_weight,
        key_weight,
        positions,
        epsilon,
        rope_theta,
        0,
        positions.stride(0),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        h_q=query_heads,
        h_k=key_heads,
        d=head_dim,
        half_d=head_dim // 2,
        round_to_input=True,
        write_cache=False,
        num_warps=4,
        num_stages=1,
    )
    return query, key


def paired_qk_norm_rope_dense_cache_(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    epsilon: float,
    rope_theta: float,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform one-token QK and stage K/V into fixed CP cache in one launch."""

    tokens, query_heads, key_heads, head_dim = _validate_qk(
        query,
        key,
        query_weight,
        key_weight,
        positions,
    )
    expected = (tokens, key_cache.shape[1], key_heads, head_dim)
    if value.shape != (tokens, key_heads, head_dim) or key_cache.shape != expected or value_cache.shape != expected:
        raise ValueError("dense CP QKV cache shape mismatch")
    if not 0 <= cache_position < key_cache.shape[1]:
        raise ValueError("dense CP cache position is invalid")
    if not (value.device == key_cache.device == value_cache.device == query.device):
        raise ValueError("dense CP QKV cache requires one CUDA device")
    if not (value.dtype == key_cache.dtype == value_cache.dtype == query.dtype):
        raise TypeError("dense CP QKV cache requires one dtype")
    _paired_qk_norm_rope_dense_kernel[(tokens, query_heads + key_heads)](
        query,
        key,
        value,
        key_cache,
        value_cache,
        query_weight,
        key_weight,
        positions,
        epsilon,
        rope_theta,
        cache_position,
        positions.stride(0),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        h_q=query_heads,
        h_k=key_heads,
        d=head_dim,
        half_d=head_dim // 2,
        round_to_input=False,
        write_cache=True,
        num_warps=4,
        num_stages=1,
    )
    return query, key


@triton.jit(do_not_specialize=["cache_length"])
def _decode_attn_nhd_short_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    cache_length,
    softmax_scale,
    sq_b,
    sq_h,
    sq_d,
    sk_b,
    sk_n,
    sk_h,
    sk_d,
    sv_b,
    sv_n,
    sv_h,
    sv_d,
    so_b,
    so_h,
    so_d,
    h_q: tl.constexpr,
    h_k: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    block_n: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    key_head = query_head // group
    offsets_d = tl.arange(0, d)
    offsets_n = tl.arange(0, block_n)
    valid = offsets_n < cache_length
    query = tl.load(q_ptr + batch * sq_b + query_head * sq_h + offsets_d * sq_d).to(tl.float32)
    query = query * softmax_scale
    key_base = k_ptr + batch * sk_b + key_head * sk_h
    keys = tl.load(
        key_base + offsets_n[:, None] * sk_n + offsets_d[None, :] * sk_d,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(keys * query[None, :], axis=1)
    scores = tl.where(valid, scores, -float("inf"))
    maximum = tl.max(scores, axis=0)
    probabilities = tl.exp(scores - maximum)
    value_base = v_ptr + batch * sv_b + key_head * sv_h
    values = tl.load(
        value_base + offsets_n[:, None] * sv_n + offsets_d[None, :] * sv_d,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    normalizer = tl.sum(probabilities, axis=0)
    result = tl.sum(probabilities[:, None] * values, axis=0) / normalizer
    destination = output_ptr + batch * so_b + query_head * so_h + offsets_d * so_d
    tl.store(destination, result.to(output_ptr.dtype.element_ty))


@torch.compiler.disable
def decode_attn_nhd_short_cache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_length: int,
) -> torch.Tensor:
    """Run fixed 16-position Code Predictor attention."""

    if not isinstance(cache_length, int) or isinstance(cache_length, bool) or not 1 <= cache_length <= 16:
        raise ValueError("short-cache attention requires cache_length in [1, 16]")
    squeeze = query.ndim == 4
    if squeeze:
        if query.shape[1] != 1:
            raise ValueError("short-cache attention supports one query token")
        query = query.squeeze(1)
    if query.ndim != 3 or key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("short-cache QKV ranks are invalid")
    rows, query_heads, head_dim = query.shape
    cache_rows, capacity, key_heads, cache_head_dim = key_cache.shape
    if (cache_rows, cache_head_dim) != (rows, head_dim) or cache_length > capacity:
        raise ValueError("short-cache QKV shapes are incompatible")
    if query_heads % key_heads or query.device.type != "cuda":
        raise ValueError("short-cache attention requires compatible CUDA heads")
    output = torch.empty_like(query)
    _decode_attn_nhd_short_cache_kernel[(rows, query_heads)](
        query,
        key_cache,
        value_cache,
        output,
        cache_length,
        1.0 / (head_dim**0.5),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        h_q=query_heads,
        h_k=key_heads,
        group=query_heads // key_heads,
        d=head_dim,
        block_n=16,
        num_warps=4,
        num_stages=2,
    )
    return output.unsqueeze(1) if squeeze else output


__all__ = [
    "decode_attn_nhd_short_cache",
    "paired_qk_norm_rope_",
    "paired_qk_norm_rope_dense_cache_",
    "paired_qk_norm_rope_paged_cache_",
]

"""Request-local token sampling used by the fixed Talker and Code Predictor jobs."""

from __future__ import annotations

import torch


def _require_rows(name: str, value: torch.Tensor, rows: int) -> None:
    if value.ndim != 1 or value.shape[0] != rows:
        raise ValueError(f"{name} must be a row vector with {rows} entries")


def _require_sampling_tensor_contract(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    repetition_penalty: torch.Tensor,
    seen_token_mask: torch.Tensor | None,
) -> None:
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    for name, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("repetition_penalty", repetition_penalty),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
    for name, value, dtype in (
        ("top_k", top_k, torch.int32),
        ("seed", seed, torch.long),
        ("offset", offset, torch.long),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    tensors = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "offset": offset,
        "repetition_penalty": repetition_penalty,
        "seen_token_mask": seen_token_mask,
    }
    mismatched = [name for name, value in tensors.items() if value is not None and value.device != logits.device]
    if mismatched:
        raise ValueError(f"logits and {', '.join(mismatched)} must share one device")
    if seen_token_mask is not None and seen_token_mask.dtype != torch.bool:
        raise TypeError("seen_token_mask must use torch.bool")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    penalty: torch.Tensor,
    seen_token_mask: torch.Tensor | None,
) -> torch.Tensor:
    if seen_token_mask is None:
        return logits
    if seen_token_mask.shape != logits.shape:
        raise ValueError("seen_token_mask must match the logits shape")
    adjusted = torch.where(logits > 0, logits / penalty.unsqueeze(1), logits * penalty.unsqueeze(1))
    return torch.where(seen_token_mask, adjusted, logits)


def _cpu_sample_row(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
    seed: int,
    offset: int,
) -> torch.Tensor:
    """Reference sampler with an isolated generator; it never advances global RNG."""

    vocab_size = logits.shape[0]
    effective_k = vocab_size if top_k <= 0 else min(top_k, vocab_size)
    values, indices = torch.topk(logits.float(), effective_k)
    probabilities = torch.softmax(values, dim=-1)
    if top_p < 1.0:
        order = torch.argsort(probabilities, descending=True)
        ordered = probabilities[order]
        remove = ordered.cumsum(dim=-1) - ordered > top_p
        ordered = ordered.masked_fill(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(0, order, ordered)
        probabilities = probabilities / probabilities.sum().clamp_min(torch.finfo(probabilities.dtype).tiny)
    generator = torch.Generator(device="cpu")
    # Offset is part of the request-local counter contract.  Mixing it into the
    # seed gives deterministic replay without a linear skip loop.
    mixed_seed = (int(seed) + 0x9E3779B97F4A7C15 * int(offset)) & ((1 << 63) - 1)
    generator.manual_seed(mixed_seed)
    sampled = torch.multinomial(probabilities.cpu(), 1, generator=generator)
    return indices.cpu().index_select(0, sampled).to(logits.device).squeeze(0)


def _validate_sampling_domain(
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    repetition_penalty: torch.Tensor,
) -> None:
    for name, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("repetition_penalty", repetition_penalty),
    ):
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
    for name, value in (
        ("temperature", temperature),
        ("top_k", top_k),
        ("seed", seed),
        ("offset", offset),
    ):
        if torch.any(value < 0):
            raise ValueError(f"{name} must be non-negative")
    if torch.any((top_p <= 0) | (top_p > 1)):
        raise ValueError("top_p must be in (0, 1]")
    if torch.any(repetition_penalty <= 0):
        raise ValueError("repetition_penalty must be positive")


def sample_logits_stateless(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    *,
    repetition_penalty: torch.Tensor | None = None,
    seen_token_mask: torch.Tensor | None = None,
    any_greedy: bool | None = None,
    any_top_k_zero: bool | None = None,
    all_top_k_zero: bool | None = None,
) -> torch.Tensor:
    """Sample one token per row from explicit seed/counter inputs.

    The optional boolean hints are accepted by captured callers but do not
    change semantics.  CUDA uses FlashInfer's deterministic stateless kernel;
    CPU is a reference implementation with private generators.
    """

    del any_greedy, any_top_k_zero, all_top_k_zero
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("logits must have shape (rows, vocabulary)")
    rows, vocab_size = logits.shape
    for name, value in (
        ("temperature", temperature),
        ("top_k", top_k),
        ("top_p", top_p),
        ("seed", seed),
        ("offset", offset),
    ):
        _require_rows(name, value, rows)
    if repetition_penalty is None:
        repetition_penalty = torch.ones_like(temperature)
    _require_rows("repetition_penalty", repetition_penalty, rows)
    _require_sampling_tensor_contract(
        logits,
        temperature,
        top_k,
        top_p,
        seed,
        offset,
        repetition_penalty,
        seen_token_mask,
    )
    _validate_sampling_domain(
        temperature,
        top_k,
        top_p,
        seed,
        offset,
        repetition_penalty,
    )

    prepared = _apply_repetition_penalty(logits, repetition_penalty, seen_token_mask)
    greedy = temperature == 0
    tokens = prepared.argmax(dim=-1).to(torch.long)
    sampled_rows = torch.nonzero(~greedy, as_tuple=False).flatten()
    if sampled_rows.numel() == 0:
        return tokens

    if logits.is_cuda:
        import flashinfer

        # FlashInfer 0.6.x assigns Philox subsequences by physical batch row.
        # Qwen3-TTS therefore samples every logical request through physical
        # row zero, keeping Talker and Code Predictor draws
        # a request's draw invariant under batching and row permutation.
        safe_temperature = torch.where(greedy, torch.ones_like(temperature), temperature)
        enabled_k = torch.where(top_k > 0, top_k.clamp(max=vocab_size), torch.full_like(top_k, vocab_size))
        safe_k = torch.where(greedy, torch.ones_like(top_k), enabled_k)
        safe_p = torch.where(greedy, torch.ones_like(top_p), top_p)
        scaled = prepared / safe_temperature.unsqueeze(1)
        return torch.cat(
            [
                flashinfer.sampling.top_k_top_p_sampling_from_logits(
                    scaled[row : row + 1],
                    safe_k[row : row + 1],
                    safe_p[row : row + 1],
                    deterministic=True,
                    seed=seed[row : row + 1],
                    offset=offset[row : row + 1],
                )
                for row in range(rows)
            ]
        ).to(torch.long)

    for row_tensor in sampled_rows:
        row = int(row_tensor.item())
        token = _cpu_sample_row(
            prepared[row] / temperature[row],
            top_k=int(top_k[row].item()),
            top_p=float(top_p[row].item()),
            seed=int(seed[row].item()),
            offset=int(offset[row].item()),
        )
        tokens[row] = token
    return tokens


__all__ = ["sample_logits_stateless"]

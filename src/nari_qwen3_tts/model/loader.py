"""Qwen3-TTS safetensors streaming and fixed packed-weight loading."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

PACKED_RULES: tuple[tuple[str, str, str | int], ...] = (
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
)
_PACKED_SHARDS = {
    target_suffix: frozenset(
        shard_id for candidate_suffix, _source_suffix, shard_id in PACKED_RULES if candidate_suffix == target_suffix
    )
    for target_suffix, _source_suffix, _shard_id in PACKED_RULES
}


def _target_and_shard(name: str) -> tuple[str, str | int | None]:
    for target_suffix, source_suffix, shard_id in PACKED_RULES:
        if source_suffix in name:
            return name.replace(source_suffix, target_suffix), shard_id
    return name, None


def _copy_weight(
    parameter: nn.Parameter,
    tensor: torch.Tensor,
    shard_id: str | int | None,
) -> None:
    custom_loader = getattr(parameter, "weight_loader", None)
    if custom_loader is not None:
        custom_loader(parameter, tensor, shard_id)
        return
    if shard_id is not None:
        raise RuntimeError(f"parameter does not accept packed shard {shard_id!r}")
    if parameter.shape != tensor.shape:
        raise ValueError(f"checkpoint shape mismatch: {tuple(parameter.shape)} != {tuple(tensor.shape)}")
    parameter.data.copy_(tensor)


@torch.no_grad()
def load_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    parameters = dict(module.named_parameters())
    loaded: set[str] = set()
    checkpoint_names: set[str] = set()
    packed_shards: dict[str, set[str | int]] = {}
    for name, tensor in weights:
        if "rotary_emb" in name:
            continue
        if name in checkpoint_names:
            raise RuntimeError(f"duplicate checkpoint weight: {name}")
        checkpoint_names.add(name)
        target, shard_id = _target_and_shard(name)
        parameter = parameters.get(target)
        if parameter is None:
            raise RuntimeError(f"unowned checkpoint weight: {name}")
        _copy_weight(parameter, tensor, shard_id)
        if shard_id is None:
            loaded.add(target)
            continue
        packed_shards.setdefault(target, set()).add(shard_id)
    for target, seen_shards in packed_shards.items():
        expected_shards = next(
            shards for suffix, shards in _PACKED_SHARDS.items() if suffix in target
        )
        if seen_shards != expected_shards:
            missing = sorted(expected_shards - seen_shards, key=str)
            raise RuntimeError(f"incomplete packed parameter {target}; missing shards: {missing}")
        loaded.add(target)
    return loaded


def _safetensors_device(device: str | torch.device) -> str:
    return "cuda" if str(device).startswith("cuda") else str(device)


def iter_safetensors_file(
    path: str | Path,
    *,
    device: str | torch.device,
    prefix: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    safe_device = _safetensors_device(device)
    with safe_open(str(path), framework="pt", device=safe_device) as handle:
        for name in handle.keys():
            if prefix is not None and not name.startswith(prefix):
                continue
            tensor = handle.get_tensor(name)
            if str(device) != safe_device:
                tensor = tensor.to(device, non_blocking=True)
            yield name, tensor


def iter_safetensors_shards(
    directory: str | Path,
    *,
    device: str | torch.device,
    prefix: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    root = Path(directory)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = [name for name in index["weight_map"] if prefix is None or name.startswith(prefix)]
        shard_names = sorted({index["weight_map"][name] for name in names})
        for shard_name in shard_names:
            yield from iter_safetensors_file(root / shard_name, device=device, prefix=prefix)
        return
    single = root / "model.safetensors"
    if single.is_file():
        yield from iter_safetensors_file(single, device=device, prefix=prefix)
        return
    raise FileNotFoundError(f"no safetensors checkpoint found in {root}")


def require_all_parameters(module: nn.Module, loaded: set[str], label: str) -> None:
    missing = set(dict(module.named_parameters())) - loaded
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        raise RuntimeError(f"missing {len(missing)} {label} parameters: {preview}")


__all__ = [
    "iter_safetensors_file",
    "iter_safetensors_shards",
    "load_hf_weights",
    "require_all_parameters",
]

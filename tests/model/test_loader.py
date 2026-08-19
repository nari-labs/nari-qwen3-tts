from __future__ import annotations

import pytest
import torch

from nari_qwen3_tts.model.layers import GatedMLP, PackedQKVLinear
from nari_qwen3_tts.model.loader import load_hf_weights, require_all_parameters


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv_proj = PackedQKVLinear(4, 2, 1, 2)
        self.mlp = GatedMLP(4, 8)


def test_loader_packs_qkv_and_gate_up_directly() -> None:
    module = _Tiny()
    weights = [
        ("attn.q_proj.weight", torch.full((4, 4), 1.0)),
        ("attn.k_proj.weight", torch.full((2, 4), 2.0)),
        ("attn.v_proj.weight", torch.full((2, 4), 3.0)),
        ("mlp.gate_proj.weight", torch.full((8, 4), 4.0)),
        ("mlp.up_proj.weight", torch.full((8, 4), 5.0)),
        ("mlp.down_proj.weight", torch.full((4, 8), 6.0)),
    ]
    loaded = load_hf_weights(module, weights)

    assert loaded == {
        "attn.qkv_proj.weight",
        "mlp.down_proj.weight",
        "mlp.gate_up_proj.weight",
    }
    assert module.attn.qkv_proj.weight[:, 0].tolist() == [1.0] * 4 + [2.0] * 2 + [3.0] * 2
    assert module.mlp.gate_up_proj.weight[:, 0].tolist() == [4.0] * 8 + [5.0] * 8


def test_loader_rejects_any_missing_parameter() -> None:
    module = _Tiny()
    loaded = set(dict(module.named_parameters()))
    loaded.remove("mlp.down_proj.weight")
    with pytest.raises(RuntimeError, match="missing 1 Tiny parameters: mlp.down_proj.weight"):
        require_all_parameters(module, loaded, "Tiny")


@pytest.mark.parametrize(
    "missing_name",
    [
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ],
)
def test_loader_rejects_partial_packed_parameters(missing_name: str) -> None:
    module = _Tiny()
    weights = [
        ("attn.q_proj.weight", torch.full((4, 4), 1.0)),
        ("attn.k_proj.weight", torch.full((2, 4), 2.0)),
        ("attn.v_proj.weight", torch.full((2, 4), 3.0)),
        ("mlp.gate_proj.weight", torch.full((8, 4), 4.0)),
        ("mlp.up_proj.weight", torch.full((8, 4), 5.0)),
        ("mlp.down_proj.weight", torch.full((4, 8), 6.0)),
    ]

    with pytest.raises(RuntimeError, match="incomplete packed parameter"):
        load_hf_weights(module, [item for item in weights if item[0] != missing_name])


@pytest.mark.parametrize(
    "duplicate_name",
    [
        "attn.q_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    ],
)
def test_loader_rejects_duplicate_checkpoint_weights(duplicate_name: str) -> None:
    module = _Tiny()
    weights = [
        ("attn.q_proj.weight", torch.full((4, 4), 1.0)),
        ("attn.k_proj.weight", torch.full((2, 4), 2.0)),
        ("attn.v_proj.weight", torch.full((2, 4), 3.0)),
        ("mlp.gate_proj.weight", torch.full((8, 4), 4.0)),
        ("mlp.up_proj.weight", torch.full((8, 4), 5.0)),
        ("mlp.down_proj.weight", torch.full((4, 8), 6.0)),
    ]
    duplicate = next(item for item in weights if item[0] == duplicate_name)

    with pytest.raises(RuntimeError, match="duplicate checkpoint weight"):
        load_hf_weights(module, [*weights, duplicate])


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("attn.q_proj.weight", (3, 4)),
        ("attn.k_proj.weight", (2, 3)),
        ("mlp.gate_proj.weight", (7, 4)),
        ("mlp.down_proj.weight", (4, 7)),
    ],
)
def test_loader_rejects_every_checkpoint_shape_mismatch(name: str, shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        load_hf_weights(_Tiny(), [(name, torch.zeros(shape))])


def test_loader_rejects_an_unowned_checkpoint_tensor() -> None:
    with pytest.raises(RuntimeError, match="unowned checkpoint weight"):
        load_hf_weights(_Tiny(), [("unexpected.weight", torch.zeros(1))])


def test_loader_ignores_only_nonpersistent_rotary_cache_tensors() -> None:
    assert load_hf_weights(
        _Tiny(),
        [("attn.rotary_emb.inv_freq", torch.zeros(2))],
    ) == set()

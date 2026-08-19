from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts.executor.optimizations import install_capture_optimizations
from nari_qwen3_tts.model.layers import (
    GatedMLP,
    RMSNorm,
    quantize_fp8_weight_blocks,
    qwen3_tts_add_rmsnorm,
    qwen3_tts_rmsnorm,
    qwen3_tts_silu_and_mul,
)


def test_fused_math_fallbacks_preserve_unfused_cpu_contract() -> None:
    gate_up = torch.randn(3, 16)
    gate, up = gate_up.chunk(2, dim=-1)
    assert torch.equal(qwen3_tts_silu_and_mul(gate_up), torch.nn.functional.silu(gate) * up)

    hidden = torch.randn(3, 8)
    residual = torch.randn(3, 8)
    norm = RMSNorm(8)
    expected_residual = residual + hidden
    expected_hidden = norm(expected_residual)
    candidate_hidden, candidate_residual = qwen3_tts_add_rmsnorm(hidden, residual, norm)
    assert torch.equal(candidate_hidden, expected_hidden)
    assert torch.equal(candidate_residual, expected_residual)
    assert torch.equal(qwen3_tts_rmsnorm(hidden, norm), norm(hidden))


def test_fp8_block_quantization_is_shape_checked_and_block_scaled() -> None:
    with pytest.raises(ValueError, match="divisible by 128"):
        quantize_fp8_weight_blocks(torch.ones((128, 64)))
    weight = torch.linspace(-2, 2, 128 * 128, dtype=torch.float32).reshape(128, 128)
    quantized, scales = quantize_fp8_weight_blocks(weight)
    assert quantized.shape == weight.shape
    assert quantized.dtype is torch.float8_e4m3fn
    assert scales.shape == (1, 1)
    assert torch.isfinite(quantized.float()).all()
    assert torch.isfinite(scales).all()


def test_mlp_keeps_checkpoint_parameter_names_and_cpu_output_without_fp8() -> None:
    mlp = GatedMLP(128, 128).eval()
    torch.nn.init.normal_(mlp.gate_up_proj.weight, std=0.02)
    torch.nn.init.normal_(mlp.down_proj.weight, std=0.02)
    values = torch.randn(2, 128)
    gate, up = mlp.gate_up_proj(values).chunk(2, dim=-1)
    expected = mlp.down_proj(torch.nn.functional.silu(gate) * up)

    assert tuple(mlp.state_dict()) == ("gate_up_proj.weight", "down_proj.weight")
    assert torch.equal(mlp(values), expected)
    assert not mlp.fp8_enabled
    assert not mlp.cuda_optimized_math_enabled
    assert mlp.initialize_fp8_gate_up_weight() is False


def test_installer_reports_fp8_coverage_without_substituting_modules() -> None:
    talker_layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    cp_layers = torch.nn.ModuleList([torch.nn.Module()])
    for layer in (*talker_layers, *cp_layers):
        layer.mlp = GatedMLP(128, 128)
    originals = [layer.mlp for layer in (*talker_layers, *cp_layers)]
    assets = SimpleNamespace(
        talker=SimpleNamespace(model=SimpleNamespace(layers=talker_layers)),
        code_predictor=SimpleNamespace(model=SimpleNamespace(layers=cp_layers, norm=RMSNorm(128))),
    )

    report = install_capture_optimizations(assets, require_talker_fp8=False)

    assert report.talker_mlp_layers == 2
    assert report.code_predictor_mlp_layers == 1
    assert report.talker_fp8_layers == 0
    # The loaded modules themselves are prepared in place; nothing is swapped.
    assert [layer.mlp for layer in (*talker_layers, *cp_layers)] == originals
    assert all(layer.mlp.cuda_optimized_math_enabled for layer in (*talker_layers, *cp_layers))


def test_installer_rejects_modules_it_does_not_own() -> None:
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Linear(4, 4)
    assets = SimpleNamespace(
        talker=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([layer]))),
        code_predictor=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList())),
    )
    with pytest.raises(TypeError, match="Qwen3-TTS GatedMLP"):
        install_capture_optimizations(assets, require_talker_fp8=False)



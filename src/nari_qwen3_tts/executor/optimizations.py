"""Post-load FP8 weight preparation for captured Qwen3-TTS execution.

Model forward methods own device- and dtype-aware kernel dispatch. This module
prepares Talker gate/up weights as FP8 blocks and enables optimized model math
after exact checkpoint loading.
"""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.model.layers import GatedMLP


@dataclass(frozen=True, slots=True)
class ExecutionOptimizationReport:
    talker_mlp_layers: int
    talker_fp8_layers: int
    code_predictor_mlp_layers: int


def install_capture_optimizations(
    assets: object,
    *,
    require_talker_fp8: bool,
) -> ExecutionOptimizationReport:
    """Quantize Talker gate/up weights and report the resulting coverage."""

    talker_layers = assets.talker.model.layers
    cp_layers = assets.code_predictor.model.layers
    for layers in (talker_layers, cp_layers):
        for layer in layers:
            if not isinstance(layer.mlp, GatedMLP):
                raise TypeError("capture optimization requires the Qwen3-TTS GatedMLP")
    fp8_layers = sum(int(layer.mlp.initialize_fp8_gate_up_weight()) for layer in talker_layers)
    if require_talker_fp8 and fp8_layers != len(talker_layers):
        raise RuntimeError(
            f"H100 profile requires FP8 gate/up for every Talker layer: {fp8_layers}/{len(talker_layers)}"
        )
    for layer in (*talker_layers, *cp_layers):
        layer.mlp.enable_cuda_optimized_math()
    return ExecutionOptimizationReport(
        talker_mlp_layers=len(talker_layers),
        talker_fp8_layers=fp8_layers,
        code_predictor_mlp_layers=len(cp_layers),
    )


__all__ = [
    "ExecutionOptimizationReport",
    "install_capture_optimizations",
]


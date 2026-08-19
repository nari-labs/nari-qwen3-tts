"""Lazy public exports for the Qwen3-TTS model."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "LoadedModelAssets": ("nari_qwen3_tts.model.checkpoint", "LoadedModelAssets"),
    "Qwen3TTSCodePredictor": ("nari_qwen3_tts.model.components", "Qwen3TTSCodePredictor"),
    "Qwen3TTSCodePredictorConfig": ("nari_qwen3_tts.model.config", "Qwen3TTSCodePredictorConfig"),
    "Qwen3TTSConfig": ("nari_qwen3_tts.model.config", "Qwen3TTSConfig"),
    "Qwen3TTSModel": ("nari_qwen3_tts.model.public", "Qwen3TTSModel"),
    "Qwen3TTSTalkerConfig": ("nari_qwen3_tts.model.config", "Qwen3TTSTalkerConfig"),
    "Qwen3TTSTalkerModel": ("nari_qwen3_tts.model.components", "Qwen3TTSTalkerModel"),
    "Qwen3TTSTextDomain": ("nari_qwen3_tts.model.text", "Qwen3TTSTextDomain"),
    "SynthesisRequest": ("nari_qwen3_tts.contract.request", "SynthesisRequest"),
    "load_model_assets": ("nari_qwen3_tts.model.checkpoint", "load_model_assets"),
    "open_model": ("nari_qwen3_tts.model.public", "open_model"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "LoadedModelAssets",
    "Qwen3TTSCodePredictor",
    "Qwen3TTSCodePredictorConfig",
    "Qwen3TTSConfig",
    "Qwen3TTSModel",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSTextDomain",
    "SynthesisRequest",
    "load_model_assets",
    "open_model",
]

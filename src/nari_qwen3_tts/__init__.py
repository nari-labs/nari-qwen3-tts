"""Nari Qwen3-TTS engine package with a lazy public model API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from nari_qwen3_tts.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, ModelAssetConfig

if TYPE_CHECKING:
    from nari_qwen3_tts.model.checkpoint import LoadedModelAssets
    from nari_qwen3_tts.model.public import Qwen3TTSModel

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "LoadedModelAssets": ("nari_qwen3_tts.model.checkpoint", "LoadedModelAssets"),
    "Qwen3TTSModel": ("nari_qwen3_tts.model.public", "Qwen3TTSModel"),
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
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "LoadedModelAssets",
    "ModelAssetConfig",
    "Qwen3TTSModel",
    "__version__",
    "load_model_assets",
    "open_model",
]

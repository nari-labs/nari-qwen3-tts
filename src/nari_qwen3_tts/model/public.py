"""Explicit public composition boundary for the one supported model."""

from __future__ import annotations

from nari_qwen3_tts.config import ModelAssetConfig
from nari_qwen3_tts.contract.model import ModelIdentityPolicy
from nari_qwen3_tts.contract.request import EncodedText, SynthesisRequest
from nari_qwen3_tts.model.capabilities import (
    QWEN3_TTS_CAPABILITIES,
    Qwen3TTSModelCapabilities,
)
from nari_qwen3_tts.model.checkpoint import CheckpointLoader, LoadedModelAssets
from nari_qwen3_tts.model.text import DEFAULT_STREAMING_TOKENIZER_POOL_SIZE, Qwen3TTSTextDomain


class Qwen3TTSModel:
    """Own model identity, tokenizer meaning, and optional raw weight loading.

    This object is the explicit public composition boundary for the one
    supported model family.
    """

    def __init__(
        self,
        config: ModelAssetConfig | None = None,
        *,
        tokenizer_pool_size: int = DEFAULT_STREAMING_TOKENIZER_POOL_SIZE,
        identity_policy: ModelIdentityPolicy | None = None,
    ) -> None:
        if type(tokenizer_pool_size) is not int:
            raise TypeError("tokenizer_pool_size must be an integer")
        if tokenizer_pool_size < 1:
            raise ValueError("tokenizer_pool_size must be positive")
        self.asset_config = config or ModelAssetConfig()
        self.checkpoint = CheckpointLoader(
            self.asset_config,
            identity_policy=identity_policy,
        )
        self.text = Qwen3TTSTextDomain(
            self.checkpoint.local_directory,
            tokenizer_pool_size=tokenizer_pool_size,
        )
        self.capabilities = QWEN3_TTS_CAPABILITIES

    @property
    def model_config(self):
        return self.checkpoint.config

    @property
    def identity(self):
        return self.checkpoint.identity

    def prepare(self, request: SynthesisRequest) -> EncodedText:
        return self.text.prepare(request)

    def prepare_live(
        self,
        request: SynthesisRequest,
        *,
        token_ids: tuple[int, ...],
        wrapped_ids: tuple[int, ...],
    ) -> EncodedText:
        return self.text.prepare_live(
            request,
            token_ids=token_ids,
            wrapped_ids=wrapped_ids,
        )

    def load_assets(self) -> LoadedModelAssets:
        assets = self.checkpoint.load_assets()
        if assets.identity != self.checkpoint.identity:
            raise RuntimeError("loaded model artifact identity differs from the resolved checkpoint")
        if assets.model_config != self.checkpoint.config:
            raise RuntimeError("loaded model config differs from the resolved checkpoint config")
        return assets


def open_model(
    config: ModelAssetConfig | None = None,
    *,
    tokenizer_pool_size: int = DEFAULT_STREAMING_TOKENIZER_POOL_SIZE,
    identity_policy: ModelIdentityPolicy | None = None,
) -> Qwen3TTSModel:
    return Qwen3TTSModel(
        config,
        tokenizer_pool_size=tokenizer_pool_size,
        identity_policy=identity_policy,
    )


__all__ = ["Qwen3TTSModel", "Qwen3TTSModelCapabilities", "open_model"]

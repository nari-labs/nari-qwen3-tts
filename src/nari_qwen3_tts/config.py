"""Public configuration for model assets and loading."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"


@dataclass(frozen=True, slots=True)
class ModelAssetConfig:
    """Inputs required to materialize the raw model assets.

    A local directory is accepted for offline operation. Remote repository
    identifiers must resolve through an immutable ``revision``. The supported
    default model uses :data:`DEFAULT_MODEL_REVISION` when no override is
    supplied; non-default remote repositories require an explicit revision.
    """

    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None
    cache_dir: str | None = None
    device: str = "cuda:0"
    require_h100: bool = True
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision must be a non-empty string when provided")
        if self.cache_dir is not None and (not isinstance(self.cache_dir, str) or not self.cache_dir.strip()):
            raise ValueError("cache_dir must be a non-empty string when provided")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if type(self.require_h100) is not bool:
            raise TypeError("require_h100 must be a boolean")
        if type(self.local_files_only) is not bool:
            raise TypeError("local_files_only must be a boolean")

    @property
    def is_local_directory(self) -> bool:
        return Path(self.model_id).is_dir()

    @property
    def effective_revision(self) -> str | None:
        if self.revision is not None:
            return self.revision
        if not self.is_local_directory and self.model_id == DEFAULT_MODEL_ID:
            return DEFAULT_MODEL_REVISION
        return None


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True, slots=True)
class LiveInputConfig:
    max_pending_text_characters: int = 65_536
    max_input_append_events: int = 1024
    max_live_text_tokens: int = 16 * 1024
    max_update_tokens: int = 4096

    def __post_init__(self) -> None:
        for name in (
            "max_pending_text_characters",
            "max_input_append_events",
            "max_live_text_tokens",
            "max_update_tokens",
        ):
            _positive_integer(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class EngineConfig:
    max_active_requests: int = 128
    command_capacity: int = 512
    max_commands_per_turn: int = 32
    idle_poll_interval_s: float = 0.0005
    max_buffered_pcm_bytes: int = 8 * 1024 * 1024
    live_input: LiveInputConfig = field(default_factory=LiveInputConfig)

    def __post_init__(self) -> None:
        for name in (
            "max_active_requests",
            "command_capacity",
            "max_commands_per_turn",
            "max_buffered_pcm_bytes",
        ):
            _positive_integer(name, getattr(self, name))
        _positive_float("idle_poll_interval_s", self.idle_poll_interval_s)
        if not isinstance(self.live_input, LiveInputConfig):
            raise TypeError("live_input must be a LiveInputConfig")


@dataclass(frozen=True, slots=True)
class ApiConfig:
    command_timeout_s: float = 10.0
    startup_timeout_s: float = 1_800.0
    shutdown_timeout_s: float = 30.0
    input_event_timeout_s: float = 30.0
    websocket_send_timeout_s: float = 5.0
    max_websocket_json_characters: int = 65_536
    transport_thread_headroom: int = 16

    def __post_init__(self) -> None:
        for name in (
            "command_timeout_s",
            "startup_timeout_s",
            "shutdown_timeout_s",
            "input_event_timeout_s",
            "websocket_send_timeout_s",
        ):
            _positive_float(name, getattr(self, name))
        for name in ("max_websocket_json_characters", "transport_thread_headroom"):
            _positive_integer(name, getattr(self, name))


__all__ = [
    "ApiConfig",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "EngineConfig",
    "LiveInputConfig",
    "ModelAssetConfig",
]

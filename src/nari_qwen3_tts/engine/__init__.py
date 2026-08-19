"""Engine-owned request mutation, claim, commit, and lifecycle mechanisms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nari_qwen3_tts.engine.engine import Engine


def __getattr__(name: str):
    if name == "Engine":
        from nari_qwen3_tts.engine.engine import Engine

        return Engine
    raise AttributeError(name)


__all__ = ["Engine"]

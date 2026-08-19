"""Import-light value contracts for append-safe Qwen3-TTS text ingress."""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.contract.errors import StreamingTextControlTokenError


@dataclass(frozen=True, slots=True)
class StreamingTextTokenization:
    token_ids: tuple[int, ...]
    stable_token_ids: tuple[int, ...]
    wrapped_ids: tuple[int, ...]
    stable_wrapped_ids: tuple[int, ...]
    stable_character_count: int


__all__ = [
    "StreamingTextControlTokenError",
    "StreamingTextTokenization",
]

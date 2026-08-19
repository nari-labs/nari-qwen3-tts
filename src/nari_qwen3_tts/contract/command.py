"""Immutable commands crossing from API callers to the engine thread."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import TypeAlias

from nari_qwen3_tts.contract.request import SynthesisRequest
from nari_qwen3_tts.contract.stream import PCMStream


def _require_request_id(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("engine command requires a non-empty request ID")


def _require_reply(value: object) -> None:
    if not isinstance(value, Future):
        raise TypeError("engine command reply must be a Future")


@dataclass(frozen=True, slots=True)
class SubmitRequest:
    request_id: str
    request: SynthesisRequest
    live: bool
    initial_token_ids: tuple[int, ...]
    initial_wrapped_ids: tuple[int, ...]
    input_finished: bool
    reply: Future[PCMStream]

    def __post_init__(self) -> None:
        _require_request_id(self.request_id)
        if not isinstance(self.request, SynthesisRequest):
            raise TypeError("submit command requires a SynthesisRequest")
        if not isinstance(self.live, bool):
            raise TypeError("submit live flag must be a boolean")
        for name, values in (
            ("initial token IDs", self.initial_token_ids),
            ("initial wrapped IDs", self.initial_wrapped_ids),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"submit {name} must be a tuple")
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"submit {name} must contain non-negative integers")
        if not isinstance(self.input_finished, bool):
            raise TypeError("submit input-finished flag must be a boolean")
        if self.live:
            if not self.initial_token_ids or len(self.initial_wrapped_ids) < 4:
                raise ValueError("live submit requires initial target and wrapped token IDs")
        elif self.initial_token_ids or self.initial_wrapped_ids or not self.input_finished:
            raise ValueError("ordinary submit cannot carry live-input token state")
        _require_reply(self.reply)


@dataclass(frozen=True, slots=True)
class AppendText:
    request_id: str
    sequence: int
    token_ids: tuple[int, ...]
    is_final: bool
    reply: Future[None]

    def __post_init__(self) -> None:
        _require_request_id(self.request_id)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("live input sequence must be a non-negative integer")
        if not isinstance(self.token_ids, tuple):
            raise TypeError("live input token IDs must be a tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
            raise ValueError("live input token IDs must be non-negative integers")
        if not isinstance(self.is_final, bool):
            raise TypeError("live input final flag must be a boolean")
        if not self.token_ids and not self.is_final:
            raise ValueError("a non-final live input update cannot be empty")
        _require_reply(self.reply)


@dataclass(frozen=True, slots=True)
class CancelRequest:
    request_id: str
    reply: Future[None]

    def __post_init__(self) -> None:
        _require_request_id(self.request_id)
        _require_reply(self.reply)


@dataclass(frozen=True, slots=True)
class StopEngine:
    reply: Future[None]

    def __post_init__(self) -> None:
        _require_reply(self.reply)


EngineCommand: TypeAlias = SubmitRequest | AppendText | CancelRequest | StopEngine


__all__ = [
    "AppendText",
    "CancelRequest",
    "EngineCommand",
    "StopEngine",
    "SubmitRequest",
]

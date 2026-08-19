"""Canonical synthesis request, encoded text, and admitted-input contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import torch

SUPPORTED_SPEAKERS = (
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian",
)
SUPPORTED_LANGUAGES = (
    "auto",
    "chinese",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
)


def _normalize_choice(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


def _strict_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _strict_int(name: str, value: object, *, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """Validated CustomVoice model inputs, independent of serving transport."""

    text: str
    voice: str = "aiden"
    language: str = "auto"
    instruct: str = ""
    non_streaming_mode: bool = True
    random_seed: int = 0
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    ignore_eos: bool = False
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    skip_fixed_bootstrap_audio: bool = True
    stream_chunk_schedule: tuple[int, ...] | None = None
    stream_chunk_frames: int | None = None
    stream_first_chunk_frames: int | None = None
    stream_steady_chunk_frames: int | None = None

    def __post_init__(self) -> None:  # noqa: PLR0912,PLR0915 - fail-closed validation is centralized
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text.strip():
            raise ValueError("text cannot be empty or whitespace")
        if not isinstance(self.instruct, str):
            raise TypeError("instruct must be a string")
        voice = _normalize_choice(self.voice, field="speaker")
        language = _normalize_choice(self.language, field="language")
        if voice not in SUPPORTED_SPEAKERS:
            raise ValueError(f"unsupported speaker: {self.voice!r}")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language: {self.language!r}")
        object.__setattr__(self, "voice", voice)
        object.__setattr__(self, "language", language)
        _strict_bool("non_streaming_mode", self.non_streaming_mode)
        _strict_bool("do_sample", self.do_sample)
        _strict_bool("ignore_eos", self.ignore_eos)
        _strict_bool("subtalker_dosample", self.subtalker_dosample)
        _strict_bool("skip_fixed_bootstrap_audio", self.skip_fixed_bootstrap_audio)
        _strict_int("random_seed", self.random_seed)
        _strict_int("top_k", self.top_k)
        _strict_int("subtalker_top_k", self.subtalker_top_k)
        _strict_int("max_new_tokens", self.max_new_tokens, minimum=1)
        for name in (
            "temperature",
            "top_p",
            "repetition_penalty",
            "subtalker_temperature",
            "subtalker_top_p",
        ):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or self.subtalker_temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 < self.subtalker_top_p <= 1:
            raise ValueError("subtalker_top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if not self.do_sample and self.temperature != 0:
            object.__setattr__(self, "temperature", 0.0)
        if self.stream_chunk_schedule is not None:
            schedule = self.stream_chunk_schedule
            if isinstance(schedule, bool) or not isinstance(schedule, (list, tuple)):
                raise TypeError("stream_chunk_schedule must be a list or tuple of integers")
            if not schedule:
                raise ValueError("stream_chunk_schedule cannot be empty")
            for value in schedule:
                _strict_int("stream_chunk_schedule entry", value, minimum=1)
            object.__setattr__(self, "stream_chunk_schedule", tuple(schedule))
        for name in (
            "stream_chunk_frames",
            "stream_first_chunk_frames",
            "stream_steady_chunk_frames",
        ):
            value = getattr(self, name)
            if value is not None:
                _strict_int(name, value, minimum=1)

    @property
    def effective_max_output_tokens(self) -> int:
        """Generation cap after the model's mandatory prefill/decode minimum."""
        return max(2, self.max_new_tokens)

    @property
    def talker_sampling(self) -> tuple[bool, float, int, float, float]:
        return (
            self.do_sample,
            float(self.temperature),
            self.top_k,
            float(self.top_p),
            float(self.repetition_penalty),
        )

    @property
    def subtalker_sampling(self) -> tuple[bool, int, float, float]:
        return (
            self.subtalker_dosample,
            self.subtalker_top_k,
            float(self.subtalker_top_p),
            float(self.subtalker_temperature),
        )

    @property
    def has_custom_stream_chunk_controls(self) -> bool:
        return any(
            value is not None
            for value in (
                self.stream_chunk_schedule,
                self.stream_chunk_frames,
                self.stream_first_chunk_frames,
                self.stream_steady_chunk_frames,
            )
        )


@dataclass(frozen=True, slots=True)
class EncodedText:
    request: SynthesisRequest
    text_token_ids: torch.Tensor
    instruct_token_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class FragmentTokenization:
    token_ids: tuple[int, ...]
    wrapped_ids: tuple[int, ...]
    consumed_character_count: int

    def __post_init__(self) -> None:
        for name, values in (
            ("token IDs", self.token_ids),
            ("wrapped IDs", self.wrapped_ids),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"fragment {name} must be a tuple")
            if any(type(token_id) is not int or token_id < 0 for token_id in values):
                raise ValueError(f"fragment {name} must contain non-negative integers")
        if (
            type(self.consumed_character_count) is not int
            or self.consumed_character_count < 0
        ):
            raise ValueError("fragment consumed character count must be a non-negative integer")
        if self.token_ids and self.consumed_character_count == 0:
            raise ValueError("fragment token IDs require consumed text")


class TextFrontend(Protocol):
    """Thread-safe CPU text preparation shared by HTTP and live ingress."""

    @property
    def streaming_tokenizer_concurrency(self) -> int: ...

    def prepare(self, request: SynthesisRequest) -> EncodedText: ...

    def prepare_live(
        self,
        request: SynthesisRequest,
        *,
        token_ids: tuple[int, ...],
        wrapped_ids: tuple[int, ...],
    ) -> EncodedText: ...

    def prepare_streaming_tokenizer_pool(self) -> None: ...

    def tokenize_streaming_fragment(
        self,
        text: str,
        *,
        is_initial: bool,
        is_final: bool,
    ) -> FragmentTokenization: ...


@dataclass(frozen=True, slots=True)
class TextContinuation:
    """Request-owned Talker text continuation, including live-update ordering."""

    non_streaming_mode: bool
    token_ids: torch.Tensor
    pad_token_id: torch.Tensor
    input_finished: bool = True
    terminal_token_id: torch.Tensor | None = None
    next_update_sequence: int = 0
    appended_token_chunks: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:  # noqa: PLR0912 - live-input validation is centralized
        if not isinstance(self.non_streaming_mode, bool):
            raise TypeError("non_streaming_mode must be a boolean")
        if not isinstance(self.input_finished, bool):
            raise TypeError("input_finished must be a boolean")
        if self.token_ids.ndim != 1:
            raise ValueError("text continuation requires target or pad tokens")
        if self.pad_token_id.ndim != 1 or self.pad_token_id.numel() != 1:
            raise ValueError("text continuation pad token must contain one ID")
        if self.token_ids.dtype is not torch.long or self.pad_token_id.dtype is not torch.long:
            raise TypeError("text continuation tokens must use torch.long")
        if self.token_ids.device != self.pad_token_id.device or self.token_ids.dtype != self.pad_token_id.dtype:
            raise ValueError("text continuation tokens must share device and dtype")
        if not isinstance(self.appended_token_chunks, tuple):
            raise TypeError("appended text token chunks must be a tuple")
        for chunk in self.appended_token_chunks:
            if not isinstance(chunk, torch.Tensor):
                raise TypeError("appended text token chunks must be tensors")
            if chunk.ndim != 1 or chunk.numel() < 1:
                raise ValueError("appended text token chunks must be non-empty vectors")
            if chunk.dtype is not self.token_ids.dtype:
                raise ValueError("appended text token chunks must share continuation dtype")
            if chunk.device.type != "cpu" and chunk.device != self.token_ids.device:
                raise ValueError("appended text token chunks must be on the host or continuation device")
        if self.input_finished and self.token_count < 1:
            raise ValueError("text continuation requires target or pad tokens")
        if isinstance(self.next_update_sequence, bool) or not isinstance(self.next_update_sequence, int):
            raise TypeError("next_update_sequence must be an integer")
        if self.next_update_sequence < 0:
            raise ValueError("next_update_sequence cannot be negative")
        if not self.input_finished:
            if self.non_streaming_mode:
                raise ValueError("live continuation requires streaming Talker input mode")
            if self.terminal_token_id is None:
                raise ValueError("unfinished continuation requires its terminal text token")
        if self.terminal_token_id is not None:
            terminal = self.terminal_token_id
            if terminal.ndim != 1 or terminal.numel() != 1:
                raise ValueError("terminal text token must contain one ID")
            if terminal.device != self.token_ids.device or terminal.dtype != self.token_ids.dtype:
                raise ValueError("terminal text token must share continuation device and dtype")

    def has_token(self, generation_step: int) -> bool:
        if isinstance(generation_step, bool) or not isinstance(generation_step, int):
            raise TypeError("generation step must be an integer")
        if generation_step < 0:
            raise ValueError("generation step must be a non-negative integer")
        return generation_step < self.token_count or self.input_finished

    @property
    def token_count(self) -> int:
        return self.token_ids.numel() + sum(
            chunk.numel() for chunk in self.appended_token_chunks
        )

    def token_at(self, generation_step: int) -> torch.Tensor:
        if not self.has_token(generation_step):
            raise RuntimeError("unfinished text continuation has no token for this generation step")
        base_tokens = self.token_ids.numel()
        if generation_step < base_tokens:
            return self.token_ids[generation_step : generation_step + 1]
        chunk_step = generation_step - base_tokens
        for chunk in self.appended_token_chunks:
            if chunk_step < chunk.numel():
                return chunk[chunk_step : chunk_step + 1]
            chunk_step -= chunk.numel()
        return self.pad_token_id

    def materialized_token_ids(self) -> torch.Tensor:
        """Return the complete logical sequence for diagnostics and tests."""
        if not self.appended_token_chunks:
            return self.token_ids
        chunks = (self.token_ids, *self.appended_token_chunks)
        if any(chunk.device != self.token_ids.device for chunk in chunks):
            chunks = tuple(chunk.detach().to(device="cpu") for chunk in chunks)
        return torch.cat(chunks)

    def append(
        self,
        token_ids: torch.Tensor,
        *,
        sequence: int,
        is_final: bool,
    ) -> TextContinuation:
        if self.input_finished:
            raise RuntimeError("text continuation is already finished")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise TypeError("live input sequence must be an integer")
        if sequence != self.next_update_sequence:
            raise ValueError(
                f"live input sequence {sequence} does not match expected {self.next_update_sequence}"
            )
        if not isinstance(is_final, bool):
            raise TypeError("is_final must be a boolean")
        if token_ids.ndim != 1:
            raise ValueError("live input tokens must be a vector")
        if token_ids.numel() == 0 and not is_final:
            raise ValueError("non-final live input cannot be empty")
        if token_ids.dtype != self.token_ids.dtype:
            raise ValueError("live input tokens must share continuation dtype")
        if token_ids.device.type != "cpu" and token_ids.device != self.token_ids.device:
            raise ValueError("live input tokens must be on the host or continuation device")
        appended_chunks = self.appended_token_chunks
        if token_ids.numel():
            appended_chunks += (token_ids,)
        if is_final:
            assert self.terminal_token_id is not None
            appended_chunks += (self.terminal_token_id,)
        return replace(
            self,
            appended_token_chunks=appended_chunks,
            input_finished=is_final,
            next_update_sequence=sequence + 1,
        )


@dataclass(frozen=True, slots=True)
class TalkerPrompt:
    text_token_ids: torch.Tensor
    codec_token_ids: torch.Tensor
    codec_token_mask: torch.Tensor
    sequence_length: int
    continuation: TextContinuation

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("Talker sequence length must be positive")
        for name in ("text_token_ids", "codec_token_ids", "codec_token_mask"):
            value = getattr(self, name)
            if value.ndim != 1 or value.numel() != self.sequence_length:
                raise ValueError(f"{name} must match the Talker sequence length")


@dataclass(frozen=True, slots=True)
class AdmittedRequest:
    request_id: str
    request: SynthesisRequest
    talker_input: TalkerPrompt
    chunk_schedule: tuple[int, ...]
    suppress_first_silent_frame: bool
    admitted_at_s: float

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request ID cannot be empty")
        if not self.chunk_schedule or any(size < 1 for size in self.chunk_schedule):
            raise ValueError("Codec chunk schedule must contain positive sizes")
        if self.talker_input.continuation.non_streaming_mode is not self.request.non_streaming_mode:
            raise ValueError("Talker plan input mode does not match the synthesis request")


__all__ = [
    "AdmittedRequest",
    "EncodedText",
    "FragmentTokenization",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_SPEAKERS",
    "SynthesisRequest",
    "TalkerPrompt",
    "TextContinuation",
]

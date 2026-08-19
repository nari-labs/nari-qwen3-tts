"""Qwen3-TTS tokenizer wrapping and append-safe text ingress."""

from __future__ import annotations

import copy
import os
import queue
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch

from nari_qwen3_tts.contract.request import (
    EncodedText,
    FragmentTokenization,
    SynthesisRequest,
)
from nari_qwen3_tts.model.streaming_text import (
    StreamingTextControlTokenError,
    StreamingTextTokenization,
)

if TYPE_CHECKING:
    from transformers import Qwen2TokenizerFast


DEFAULT_STREAMING_TOKENIZER_POOL_SIZE = min(4, max(1, os.cpu_count() or 1))


def _strict_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")

class Qwen3TTSTextDomain:
    """Own exact CustomVoice tokenizer wrapping and append-safe text tokenization."""

    _ASSISTANT_PREFIX = "<|im_start|>assistant\n"
    _ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

    def __init__(
        self,
        model_directory: str | Path,
        *,
        tokenizer_pool_size: int = DEFAULT_STREAMING_TOKENIZER_POOL_SIZE,
    ) -> None:
        if type(tokenizer_pool_size) is not int or tokenizer_pool_size < 1:
            raise ValueError("tokenizer_pool_size must be a positive integer")
        from transformers import Qwen2TokenizerFast

        self.model_directory = Path(model_directory).resolve()
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.model_directory,
            fix_mistral_regex=True,
        )
        self._tokenizer_pool_size = tokenizer_pool_size
        self._tokenizer_lock = threading.RLock()
        self._pool_init_lock = threading.Lock()
        self._pool: queue.LifoQueue[Qwen2TokenizerFast] | None = None
        self._forbidden_control_tokens = tuple(
            sorted(
                {
                    *(token for token in self.tokenizer.all_special_tokens if token),
                    *(str(token) for token in self.tokenizer.added_tokens_decoder.values() if str(token)),
                },
                key=len,
                reverse=True,
            )
        )
        self._assistant_prefix_ids = self.tokenizer(
            self._ASSISTANT_PREFIX,
            add_special_tokens=False,
        )["input_ids"]
        self._assistant_suffix_ids = self.tokenizer(
            self._ASSISTANT_SUFFIX,
            add_special_tokens=False,
        )["input_ids"]

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        for name in ("_tokenizer_lock", "_pool_init_lock", "_pool"):
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._tokenizer_lock = threading.RLock()
        self._pool_init_lock = threading.Lock()
        self._pool = None

    def prepare_streaming_tokenizer_pool(self) -> None:
        if self._pool is not None:
            return
        with self._pool_init_lock:
            if self._pool is not None:
                return
            pool: queue.LifoQueue[Qwen2TokenizerFast] = queue.LifoQueue(maxsize=self._tokenizer_pool_size)
            with self._tokenizer_lock:
                for _ in range(self._tokenizer_pool_size):
                    pool.put(copy.deepcopy(self.tokenizer))
            self._pool = pool

    @contextmanager
    def _tokenizer_lease(self) -> Iterator[Qwen2TokenizerFast]:
        self.prepare_streaming_tokenizer_pool()
        assert self._pool is not None
        tokenizer = self._pool.get()
        try:
            yield tokenizer
        finally:
            self._pool.put(tokenizer)

    @property
    def streaming_tokenizer_concurrency(self) -> int:
        return self._tokenizer_pool_size

    def prepare(self, request: SynthesisRequest) -> EncodedText:
        assistant_text = f"{self._ASSISTANT_PREFIX}{request.text}{self._ASSISTANT_SUFFIX}"
        with self._tokenizer_lock:
            text_ids = self.tokenizer(assistant_text, return_tensors="pt")["input_ids"][0].to(torch.long)
            if request.instruct:
                instruct_text = f"<|im_start|>user\n{request.instruct}<|im_end|>\n"
                instruct_ids = self.tokenizer(instruct_text, return_tensors="pt")["input_ids"][0].to(torch.long)
            else:
                instruct_ids = torch.empty(0, dtype=torch.long)
        return EncodedText(
            request=request,
            text_token_ids=text_ids,
            instruct_token_ids=instruct_ids,
        )

    def prepare_live(
        self,
        request: SynthesisRequest,
        *,
        token_ids: tuple[int, ...],
        wrapped_ids: tuple[int, ...],
    ) -> EncodedText:
        if not isinstance(request, SynthesisRequest):
            raise TypeError("live text preparation requires a SynthesisRequest")
        for name, values in (("target", token_ids), ("wrapped", wrapped_ids)):
            if not isinstance(values, tuple) or any(
                type(token_id) is not int or token_id < 0 for token_id in values
            ):
                raise ValueError(f"live {name} token IDs must be non-negative integers")
        prefix, suffix = self._wrapper_ids()
        if (
            len(wrapped_ids) < 3 + len(suffix)
            or tuple(wrapped_ids[:3]) != tuple(prefix[:3])
            or tuple(wrapped_ids[-len(suffix) :]) != tuple(suffix)
        ):
            raise ValueError("live wrapped token IDs do not preserve the assistant wrapper")
        if tuple(wrapped_ids[3 : -len(suffix)]) != token_ids:
            raise ValueError("live target token IDs do not match the wrapped prompt")
        if request.instruct:
            instruct_text = f"<|im_start|>user\n{request.instruct}<|im_end|>\n"
            with self._tokenizer_lock:
                instruct_ids = self.tokenizer(instruct_text, return_tensors="pt")["input_ids"][0].to(torch.long)
        else:
            instruct_ids = torch.empty(0, dtype=torch.long)
        return EncodedText(
            request=request,
            text_token_ids=torch.tensor(wrapped_ids, dtype=torch.long),
            instruct_token_ids=instruct_ids,
        )

    def _wrapper_ids(self) -> tuple[list[int], list[int]]:
        return list(self._assistant_prefix_ids), list(self._assistant_suffix_ids)

    def _tokenize_wrapped_with(self, tokenizer: Qwen2TokenizerFast, text: str) -> list[int]:
        prefix, suffix = self._wrapper_ids()
        wrapped = list(
            tokenizer(
                f"{self._ASSISTANT_PREFIX}{text}{self._ASSISTANT_SUFFIX}",
                add_special_tokens=False,
            )["input_ids"]
        )
        if len(wrapped) < 2 + len(suffix) or wrapped[:2] != prefix[:2] or wrapped[-len(suffix) :] != suffix:
            raise RuntimeError("Qwen3-TTS prompt wrapper tokenization changed")
        return wrapped

    def _validate_target_control_tokens(self, text: str) -> int:
        unstable_suffix_start = len(text)
        for token in self._forbidden_control_tokens:
            if token in text:
                raise StreamingTextControlTokenError(
                    f"Target text must not contain reserved tokenizer control token {token!r}"
                )
            for prefix_length in range(min(len(text), len(token) - 1), 0, -1):
                if text.endswith(token[:prefix_length]):
                    unstable_suffix_start = min(unstable_suffix_start, len(text) - prefix_length)
                    break
        return unstable_suffix_start

    def tokenize_streaming_fragment(
        self,
        text: str,
        *,
        is_initial: bool,
        is_final: bool,
    ) -> FragmentTokenization:
        if not isinstance(text, str):
            raise TypeError("streaming text must be a string")
        _strict_bool("is_initial", is_initial)
        _strict_bool("is_final", is_final)
        if not text:
            return FragmentTokenization((), (), 0)
        suffix_length = len(self._assistant_suffix_ids)
        with self._tokenizer_lease() as tokenizer:
            stable_special_end = self._validate_target_control_tokens(text)
            if is_initial:
                wrapped = self._tokenize_wrapped_with(tokenizer, text)
                token_ids = wrapped[3:-suffix_length]
            else:
                wrapped = []
                token_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            if is_final:
                return FragmentTokenization(tuple(token_ids), tuple(wrapped), len(text))
            pieces = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            boundaries = [int(piece[1][1]) for piece in pieces[:-1] if 0 < int(piece[1][1]) <= stable_special_end]
            for boundary in reversed(boundaries):
                candidate = text[:boundary]
                if is_initial:
                    candidate_wrapped = self._tokenize_wrapped_with(tokenizer, candidate)
                    candidate_ids = candidate_wrapped[3:-suffix_length]
                    prefix_stable = candidate_wrapped[:3] == wrapped[:3]
                else:
                    candidate_wrapped = []
                    candidate_ids = list(tokenizer(candidate, add_special_tokens=False)["input_ids"])
                    prefix_stable = True
                if candidate_ids and prefix_stable and token_ids[: len(candidate_ids)] == candidate_ids:
                    return FragmentTokenization(
                        tuple(candidate_ids),
                        tuple(candidate_wrapped),
                        boundary,
                    )
        return FragmentTokenization((), (), 0)

    def tokenize_streaming(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("streaming text must be a string")
        with self._tokenizer_lock:
            wrapped = self._tokenize_wrapped_with(self.tokenizer, text)
        return wrapped[3 : -len(self._assistant_suffix_ids)]

    def tokenize_streaming_state(self, text: str) -> StreamingTextTokenization:
        if not isinstance(text, str):
            raise TypeError("streaming text must be a string")
        with self._tokenizer_lock:
            wrapped = self._tokenize_wrapped_with(self.tokenizer, text)
            suffix_length = len(self._assistant_suffix_ids)
            token_ids = wrapped[3:-suffix_length]
            pieces = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            boundaries = [int(piece[1][1]) for piece in pieces[:-1]]
            stable_character_count = 0
            stable_wrapped = self._tokenize_wrapped_with(self.tokenizer, "")
            stable_token_ids: list[int] = []
            for boundary in reversed(boundaries):
                candidate_wrapped = self._tokenize_wrapped_with(self.tokenizer, text[:boundary])
                candidate_ids = candidate_wrapped[3:-suffix_length]
                if token_ids[: len(candidate_ids)] == candidate_ids:
                    stable_character_count = boundary
                    stable_wrapped = candidate_wrapped
                    stable_token_ids = candidate_ids
                    break
        return StreamingTextTokenization(
            token_ids=tuple(token_ids),
            stable_token_ids=tuple(stable_token_ids),
            wrapped_ids=tuple(wrapped),
            stable_wrapped_ids=tuple(stable_wrapped),
            stable_character_count=stable_character_count,
        )


__all__ = [
    "DEFAULT_STREAMING_TOKENIZER_POOL_SIZE",
    "EncodedText",
    "Qwen3TTSTextDomain",
]

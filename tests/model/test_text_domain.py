from __future__ import annotations

import copy
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts.contract.request import SynthesisRequest
from nari_qwen3_tts.model.streaming_text import StreamingTextControlTokenError
from nari_qwen3_tts.model.text import Qwen3TTSTextDomain


class _FakePreTokenizer:
    @staticmethod
    def pre_tokenize_str(text: str):
        pieces = []
        start = 0
        for index, character in enumerate(text):
            if character.isspace() and index > start:
                pieces.append((text[start:index], (start, index)))
                start = index
        if start < len(text):
            pieces.append((text[start:], (start, len(text))))
        return pieces


class _FakeTokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]
    added_tokens_decoder = {1: "<|im_start|>", 2: "<|im_end|>"}

    def __init__(self) -> None:
        self.backend_tokenizer = SimpleNamespace(pre_tokenizer=_FakePreTokenizer())

    def __call__(self, text: str, *, return_tensors: str | None = None, add_special_tokens: bool = True):
        del add_special_tokens
        ids = [ord(character) for character in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class _TokenizerActivity:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _StructuredFakeTokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_end|>", "<|custom|>"]
    added_tokens_decoder = {
        101: "<|im_start|>",
        104: "<|im_end|>",
        105: "<|custom|>",
    }
    _prefix = "<|im_start|>assistant\n"
    _suffix = "<|im_end|>\n<|im_start|>assistant\n"
    _prefix_ids = [101, 102, 103]
    _suffix_ids = [104, 103, 101, 102, 103]

    def __init__(self, activity: _TokenizerActivity) -> None:
        self.activity = activity
        self.backend_tokenizer = SimpleNamespace(pre_tokenizer=_FakePreTokenizer())

    def __deepcopy__(self, _memo):
        return type(self)(self.activity)

    @staticmethod
    def _plain(text: str) -> list[int]:
        return [1_000 + ord(character) for character in text]

    def __call__(self, text: str, *, return_tensors: str | None = None, add_special_tokens: bool = True):
        del add_special_tokens
        self.activity.enter()
        try:
            time.sleep(0.005)
            if "explode" in text:
                raise RuntimeError("tokenizer failure")
            if text == self._prefix:
                ids = list(self._prefix_ids)
            elif text == self._suffix:
                ids = list(self._suffix_ids)
            elif text.startswith(self._prefix) and text.endswith(self._suffix):
                body = text[len(self._prefix) : -len(self._suffix)]
                ids = [*self._prefix_ids, *self._plain(body), *self._suffix_ids]
            else:
                ids = self._plain(text)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return {"input_ids": ids}
        finally:
            self.activity.leave()


def _domain(monkeypatch, tmp_path, *, pool_size: int = 2) -> Qwen3TTSTextDomain:
    monkeypatch.setattr(
        "transformers.Qwen2TokenizerFast.from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    return Qwen3TTSTextDomain(tmp_path, tokenizer_pool_size=pool_size)


def _structured_domain(
    monkeypatch,
    tmp_path,
    *,
    pool_size: int = 3,
) -> tuple[Qwen3TTSTextDomain, _TokenizerActivity]:
    activity = _TokenizerActivity()
    monkeypatch.setattr(
        "transformers.Qwen2TokenizerFast.from_pretrained",
        lambda *_args, **_kwargs: _StructuredFakeTokenizer(activity),
    )
    return Qwen3TTSTextDomain(tmp_path, tokenizer_pool_size=pool_size), activity


def test_tokenizer_is_constructed_from_the_owned_source_with_exact_compatibility_flag(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def from_pretrained(source, **kwargs):
        calls.append((source, kwargs))
        return _FakeTokenizer()

    monkeypatch.setattr("transformers.Qwen2TokenizerFast.from_pretrained", from_pretrained)
    domain = Qwen3TTSTextDomain(tmp_path, tokenizer_pool_size=2)

    assert domain.model_directory == tmp_path.resolve()
    assert calls == [(tmp_path.resolve(), {"fix_mistral_regex": True})]


def test_model_owns_text_and_nonempty_instruction_wrappers(monkeypatch, tmp_path) -> None:
    domain = _domain(monkeypatch, tmp_path)
    plain = domain.prepare(SynthesisRequest(text="hello"))
    instructed = domain.prepare(SynthesisRequest(text="hello", instruct="Speak clearly."))

    assert plain.instruct_token_ids.numel() == 0
    assert instructed.instruct_token_ids.tolist() == [
        ord(character) for character in "<|im_start|>user\nSpeak clearly.<|im_end|>\n"
    ]
    assert instructed.text_token_ids.tolist() == [
        ord(character)
        for character in "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"
    ]
    assert plain.request.text == "hello"
    assert plain.text_token_ids.ndim == plain.instruct_token_ids.ndim == 1
    assert plain.text_token_ids.dtype == plain.instruct_token_ids.dtype == torch.long
    assert plain.text_token_ids.device.type == plain.instruct_token_ids.device.type == "cpu"


def test_live_prepare_reuses_initial_wrapped_ids_without_retokenizing_target(
    monkeypatch,
    tmp_path,
) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path)
    request = SynthesisRequest(text="hello")
    fragment = domain.tokenize_streaming_fragment(
        request.text,
        is_initial=True,
        is_final=True,
    )

    class _RejectRetokenization:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("initial live target text must not be tokenized twice")

    domain.tokenizer = _RejectRetokenization()
    prepared = domain.prepare_live(
        request,
        token_ids=fragment.token_ids,
        wrapped_ids=fragment.wrapped_ids,
    )

    assert prepared.request is request
    assert prepared.text_token_ids.tolist() == list(fragment.wrapped_ids)
    assert prepared.instruct_token_ids.numel() == 0


def test_tokenizer_pool_is_bounded_independent_and_restored_after_pickle(monkeypatch, tmp_path) -> None:
    domain = _domain(monkeypatch, tmp_path, pool_size=3)
    assert domain.streaming_tokenizer_concurrency == 3
    domain.prepare_streaming_tokenizer_pool()
    assert domain._pool is not None
    clones = list(domain._pool.queue)
    assert len(clones) == 3
    assert len({id(tokenizer) for tokenizer in clones}) == 3
    assert all(tokenizer is not domain.tokenizer for tokenizer in clones)

    restored = pickle.loads(pickle.dumps(domain))
    deepcopied = copy.deepcopy(domain)
    for candidate in (restored, deepcopied):
        assert candidate._pool is None
        assert candidate.streaming_tokenizer_concurrency == 3
        assert candidate.tokenize_streaming("hello world") == domain.tokenize_streaming("hello world")
        candidate.prepare_streaming_tokenizer_pool()
        assert candidate._pool is not None
        assert candidate._pool.qsize() == 3


def test_live_fragments_concatenate_to_exact_whole_text_tokens(monkeypatch, tmp_path) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path)
    text = "alpha beta gamma"
    remaining = text
    token_ids: list[int] = []
    initial = True

    while remaining:
        result = domain.tokenize_streaming_fragment(
            remaining,
            is_initial=initial,
            is_final=False,
        )
        if result.consumed_character_count == 0:
            break
        token_ids.extend(result.token_ids)
        remaining = remaining[result.consumed_character_count :]
        initial = False
    terminal = domain.tokenize_streaming_fragment(
        remaining,
        is_initial=initial,
        is_final=True,
    )
    token_ids.extend(terminal.token_ids)

    assert terminal.consumed_character_count == len(remaining)
    assert token_ids == domain.tokenize_streaming(text)


@pytest.mark.parametrize("token", _StructuredFakeTokenizer.all_special_tokens)
def test_every_reserved_control_token_is_rejected_at_fragment_ingress(
    monkeypatch,
    tmp_path,
    token: str,
) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path)
    with pytest.raises(StreamingTextControlTokenError, match="reserved"):
        domain.tokenize_streaming_fragment(
            f"target {token} text",
            is_initial=True,
            is_final=False,
        )


@pytest.mark.parametrize("token", _StructuredFakeTokenizer.all_special_tokens)
def test_whole_live_text_preserves_control_token_compatibility(
    monkeypatch,
    tmp_path,
    token: str,
) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path)
    assert domain.tokenize_streaming(f"target {token} text")


@pytest.mark.parametrize("token", _StructuredFakeTokenizer.all_special_tokens)
def test_partial_reserved_control_suffix_is_withheld(monkeypatch, tmp_path, token: str) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path)
    stable = "safe "
    partial = token[:-1]
    result = domain.tokenize_streaming_fragment(
        stable + partial,
        is_initial=True,
        is_final=False,
    )

    assert result.consumed_character_count <= len(stable)
    assert all(character_id not in result.token_ids for character_id in (101, 104, 105))


def test_streaming_tokenizer_pool_initialization_is_idempotent_under_concurrency(
    monkeypatch,
    tmp_path,
) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path, pool_size=3)

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(lambda _index: domain.prepare_streaming_tokenizer_pool(), range(24)))

    assert domain._pool is not None
    assert domain._pool.qsize() == 3
    assert len({id(tokenizer) for tokenizer in domain._pool.queue}) == 3


@pytest.mark.parametrize(
    ("text", "is_initial", "is_final", "error"),
    [
        (1, True, False, TypeError),
        ("text", 1, False, TypeError),
        ("text", True, 0, TypeError),
    ],
)
def test_live_fragment_types_fail_closed(
    monkeypatch,
    tmp_path,
    text,
    is_initial,
    is_final,
    error,
) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path)
    with pytest.raises(error):
        domain.tokenize_streaming_fragment(
            text,
            is_initial=is_initial,
            is_final=is_final,
        )


def test_concurrent_fragment_tokenization_uses_bounded_independent_leases(
    monkeypatch,
    tmp_path,
) -> None:
    domain, activity = _structured_domain(monkeypatch, tmp_path, pool_size=3)
    activity.maximum = 0
    inputs = [f"parallel fragment {index}" for index in range(9)]

    with ThreadPoolExecutor(max_workers=9) as executor:
        results = list(
            executor.map(
                lambda text: domain.tokenize_streaming_fragment(
                    text,
                    is_initial=True,
                    is_final=True,
                ),
                inputs,
            )
        )

    assert activity.maximum == 3
    assert [result.consumed_character_count for result in results] == [len(text) for text in inputs]
    assert domain._pool is not None
    assert domain._pool.qsize() == 3


def test_tokenizer_exception_returns_fragment_lease_to_pool(monkeypatch, tmp_path) -> None:
    domain, _activity = _structured_domain(monkeypatch, tmp_path, pool_size=2)
    with pytest.raises(RuntimeError, match="tokenizer failure"):
        domain.tokenize_streaming_fragment(
            "explode",
            is_initial=True,
            is_final=True,
        )

    assert domain._pool is not None
    assert domain._pool.qsize() == 2

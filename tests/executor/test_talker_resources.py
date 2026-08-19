from __future__ import annotations

from types import SimpleNamespace

import torch

from nari_qwen3_tts.executor.talker import TalkerExecutor


class _Cache:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []

    def add_request(self, request_id: str) -> None:
        self.added.append(request_id)

    def remove_request(self, request_id: str) -> None:
        self.removed.append(request_id)


class _Model:
    def __init__(self) -> None:
        self.initialized = 0
        self.device = torch.device("cpu")
        self._embedding = torch.nn.Embedding(32, 8)

    def initialize_projected_text_embedding_cache(self) -> None:
        self.initialized += 1

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self._embedding


def _config() -> object:
    predictor = SimpleNamespace(vocab_size=16)
    talker = SimpleNamespace(
        hidden_size=8,
        vocab_size=32,
        codec_eos_token_id=15,
        code_predictor=predictor,
    )
    return SimpleNamespace(talker=talker, talker_config=talker)


def test_talker_executor_owns_model_cache_and_input_planning() -> None:
    model = _Model()
    cache = _Cache()
    config = _config()
    executor = TalkerExecutor(
        model=model,
        config=config,
        cache=cache,
        capture_slots=1,
        driver=object(),
    )

    assert executor.model is model
    assert executor.config is config
    assert executor.cache is cache
    assert model.initialized == 1
    assert callable(executor.prepare_prepared_inputs)

    executor.add_request("request-a")
    executor.remove_request("request-a")
    assert cache.added == ["request-a"]
    assert cache.removed == ["request-a"]

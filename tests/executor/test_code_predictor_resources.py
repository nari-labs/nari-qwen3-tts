from __future__ import annotations

from types import SimpleNamespace

import torch

from nari_qwen3_tts.executor.code_predictor import CodePredictorExecutor


def test_code_predictor_executor_owns_model_fixed_kv_and_sampling() -> None:
    model = SimpleNamespace(
        small_to_mtp_projection=torch.nn.Linear(8, 4, bias=False),
    )
    layer0 = torch.nn.Embedding(16, 8)
    predictor = SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
    )
    config = SimpleNamespace(
        num_code_groups=16,
        talker=SimpleNamespace(hidden_size=8),
        code_predictor=predictor,
    )
    executor = CodePredictorExecutor(
        model=model,
        layer0_embedding=layer0,
        config=config,
        max_batch_size=8,
        driver=object(),
    )

    assert executor.model is model
    assert executor.layer0_embedding is layer0
    assert executor.config is config
    assert executor.max_batch_size == 8
    assert callable(executor.whole_frame)

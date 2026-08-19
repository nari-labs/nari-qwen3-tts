from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import torch

from nari_qwen3_tts.planner import CaptureCatalog
from nari_qwen3_tts.profile import (
    BatchCaptureConfig,
    CodecBatchCaptureConfig,
    CodecCaptureConfig,
    ExecutionProfile,
    PrefillCaptureConfig,
    ProfileLoader,
)


def test_factory_scratch_pages_cover_the_largest_prefill_token_bucket(monkeypatch) -> None:
    factory = importlib.import_module("nari_qwen3_tts.executor.build")
    captured_cache_args: dict[str, object] = {}

    class FakeCache:
        def __init__(self, **kwargs) -> None:
            captured_cache_args.update(kwargs)

    class FakeComponent:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class FakePool:
        @staticmethod
        def graph_pool_handle():
            return object()

    monkeypatch.setattr(factory, "PagedTalkerKV", FakeCache)
    monkeypatch.setattr(factory, "TalkerExecutor", FakeComponent)
    monkeypatch.setattr(factory, "CodePredictorExecutor", FakeComponent)
    monkeypatch.setattr(factory, "CodecExecutor", FakeComponent)
    monkeypatch.setattr(factory, "TorchCaptureDriver", FakeComponent)
    monkeypatch.setattr(factory, "CudaGraphPoolFence", FakeComponent)
    monkeypatch.setattr(factory, "install_capture_optimizations", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "graphs", FakePool())

    base = ProfileLoader().load_profile(ExecutionProfile.TTFA)
    config = replace(
        base,
        stages=replace(
            base.stages,
            talker_prefill=PrefillCaptureConfig(
                max_batch_size=1,
                batch_sizes=(1,),
                token_buckets=(1024,),
                exact_sequence_lengths=(10,),
                exact_batch_sizes=(1,),
            ),
            talker_decode=BatchCaptureConfig(1, (1,)),
            code_predictor=BatchCaptureConfig(1, (1,)),
            codec=CodecCaptureConfig(
                max_batch_size=1,
                chunk_schedule=(1,),
                suppressed_bootstrap_chunk_schedule=(1,),
                batches=CodecBatchCaptureConfig(
                    whole_sequence_first_frame=(1,),
                    whole_sequence_followup=(1,),
                    cold=(1,),
                    warm_partial=(1,),
                    warm_full=(1,),
                ),
            ),
        ),
        resources=replace(
            base.resources,
            kv_pages=32,
            kv_page_size=128,
            workspace_bytes=1024,
        ),
    )
    talker_config = SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=2,
        num_attention_heads=4,
        head_dim=8,
        hidden_size=8,
        vocab_size=32,
    )
    weight = torch.empty((1,), dtype=torch.float32)
    assets = SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(talker=talker_config, num_code_groups=16),
        talker=SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=weight)),
        code_predictor=object(),
        code_predictor_layer0_embedding=object(),
        codec=SimpleNamespace(
            decoder=SimpleNamespace(
                pre_transformer=SimpleNamespace(config=SimpleNamespace(sliding_window=9)),
            ),
        ),
    )

    catalog = CaptureCatalog.from_config(config.stages)
    factory.build_cuda_execution(
        assets,
        config=config,
        required_keys=catalog.required_keys,
    )

    assert captured_cache_args["scratch_page_count"] == 8

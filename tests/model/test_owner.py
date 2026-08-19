from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts import ModelAssetConfig
from nari_qwen3_tts.model.checkpoint import CheckpointLoader, _validate_device


def test_h100_guard_rejects_cpu_but_explicit_development_override_allows_it() -> None:
    with pytest.raises(RuntimeError, match="require_h100=True"):
        _validate_device(ModelAssetConfig(device="cpu"), torch.device("cpu"))

    _validate_device(
        ModelAssetConfig(device="cpu", require_h100=False),
        torch.device("cpu"),
    )


def test_h100_guard_rejects_unavailable_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _validate_device(ModelAssetConfig(), torch.device("cuda:0"))


def test_h100_guard_rejects_a_different_cuda_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA A100-SXM4-80GB")
    with pytest.raises(RuntimeError, match="requires H100"):
        _validate_device(ModelAssetConfig(), torch.device("cuda:0"))


def test_codec_loader_uses_the_owned_snapshot_component_and_qualified_mode(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    codec_model = torch.nn.Linear(2, 2)
    codec_model.train()

    def fake_from_pretrained(_cls, path: str, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(model=codec_model)

    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    monkeypatch.setattr(Qwen3TTSTokenizer, "from_pretrained", classmethod(fake_from_pretrained))
    checkpoint = object.__new__(CheckpointLoader)
    checkpoint.local_directory = tmp_path

    loaded = checkpoint.load_codec(device=torch.device("cpu"), dtype=torch.bfloat16)

    assert loaded is codec_model
    assert loaded.training is False
    assert calls == [
        (
            str(tmp_path / "speech_tokenizer"),
            {"device_map": "cpu", "dtype": torch.bfloat16},
        )
    ]

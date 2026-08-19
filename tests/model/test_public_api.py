from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from conftest import SRC_ROOT

from nari_qwen3_tts import ModelAssetConfig
from nari_qwen3_tts.model.public import Qwen3TTSModel


def test_package_import_is_lazy_and_does_not_construct_model_runtime() -> None:
    code = """
import json
import sys
import nari_qwen3_tts
print(json.dumps({
    "checkpoint": "nari_qwen3_tts.model.checkpoint" in sys.modules,
    "text": "nari_qwen3_tts.model.text" in sys.modules,
    "torch": "torch" in sys.modules,
    "transformers": "transformers" in sys.modules,
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "checkpoint": False,
        "text": False,
        "torch": False,
        "transformers": False,
    }


@pytest.mark.parametrize(
    ("loaded_source", "loaded_config", "message"),
    [
        ("source-b", "config-a", "identity"),
        ("source-a", "config-b", "config"),
    ],
)
def test_public_model_rejects_identity_or_config_drift_before_returning_assets(
    loaded_source: str,
    loaded_config: str,
    message: str,
) -> None:
    model = object.__new__(Qwen3TTSModel)
    model.asset_config = ModelAssetConfig(device="cpu", require_h100=False)
    assets = SimpleNamespace(identity=loaded_source, model_config=loaded_config)
    model.checkpoint = SimpleNamespace(
        identity="source-a",
        config="config-a",
        load_assets=lambda: assets,
    )

    with pytest.raises(RuntimeError, match=message):
        model.load_assets()


def test_public_model_returns_assets_only_when_composition_identity_is_stable() -> None:
    model = object.__new__(Qwen3TTSModel)
    model.asset_config = ModelAssetConfig(device="cpu", require_h100=False)
    assets = SimpleNamespace(identity="source-a", model_config="config-a")
    model.checkpoint = SimpleNamespace(
        identity="source-a",
        config="config-a",
        load_assets=lambda: assets,
    )

    assert model.load_assets() is assets


def test_public_model_loads_assets_from_its_resolved_identity() -> None:
    assets = SimpleNamespace(identity="source-a", model_config="config-a")
    calls = []

    class ExistingCheckpoint:
        identity = "source-a"
        config = "config-a"

        @staticmethod
        def load_assets():
            calls.append("load_assets")
            return assets

    model = object.__new__(Qwen3TTSModel)
    model.asset_config = ModelAssetConfig(device="cpu", require_h100=False)
    model.checkpoint = ExistingCheckpoint()

    assert model.load_assets() is assets
    assert calls == ["load_assets"]


@pytest.mark.parametrize("pool_size", [True, 0, -1, 1.5])
def test_public_model_rejects_invalid_tokenizer_pool_before_resolving_assets(
    monkeypatch,
    pool_size: object,
) -> None:
    monkeypatch.setattr(
        "nari_qwen3_tts.model.public.CheckpointLoader",
        lambda _config: pytest.fail("invalid public input must fail before model resolution"),
    )
    with pytest.raises((TypeError, ValueError), match="tokenizer_pool_size"):
        Qwen3TTSModel(tokenizer_pool_size=pool_size)  # type: ignore[arg-type]

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nari_qwen3_tts import ModelAssetConfig
from nari_qwen3_tts.contract.model import (
    FileDigest,
    ModelArtifactIdentity,
    ModelIdentityPolicy,
)
from nari_qwen3_tts.model.checkpoint import (
    _configure_model_math,
    build_model_artifact_identity,
    resolve_model_directory,
)

VALIDATED_MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"


def _identity(*, revision: str | None = "revision-one", manifest: str = "a" * 64):
    return ModelArtifactIdentity(
        requested_model_id="test/model",
        requested_revision=revision,
        resolved_directory="/tmp/model",
        resolved_revision=revision,
        files=(FileDigest("config.json", 2, "b" * 64),),
        manifest_sha256=manifest,
    )


def test_model_identity_policy_checks_only_supplied_expectations() -> None:
    identity = _identity()

    ModelIdentityPolicy().validate(identity)
    ModelIdentityPolicy(expected_revision="revision-one").validate(identity)
    ModelIdentityPolicy(expected_manifest_sha256="a" * 64).validate(identity)

    with pytest.raises(RuntimeError, match="revision"):
        ModelIdentityPolicy(expected_revision="revision-two").validate(identity)
    with pytest.raises(RuntimeError, match="manifest"):
        ModelIdentityPolicy(expected_manifest_sha256="c" * 64).validate(identity)


def _write_codec_source(root: Path) -> None:
    codec = root / "speech_tokenizer"
    codec.mkdir()
    (codec / "config.json").write_text(json.dumps({"model_type": "qwen3_tts_tokenizer_12hz"}))
    (codec / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 24_000}))
    (codec / "model.safetensors").write_bytes(b"codec-checkpoint")


def test_model_math_uses_the_validated_float32_matmul_contract() -> None:
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        _configure_model_math()
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("model_id", "", ValueError),
        ("revision", " ", ValueError),
        ("cache_dir", "", ValueError),
        ("device", "", ValueError),
        ("require_h100", 1, TypeError),
        ("local_files_only", 0, TypeError),
    ],
)
def test_asset_config_is_fail_closed(field, value, error) -> None:
    with pytest.raises(error):
        ModelAssetConfig(**{field: value})


def test_source_manifest_hashes_model_and_tokenizer_content(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model": "one"}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"type": "qwen2"}))
    (tmp_path / "vocab.json").write_text(json.dumps({"tokenizer": "one"}))
    (tmp_path / "merges.txt").write_text("merge one")
    (tmp_path / "model.safetensors").write_bytes(b"checkpoint-one")
    _write_codec_source(tmp_path)
    first = build_model_artifact_identity(
        tmp_path,
        requested_model_id="local-test",
        requested_revision="revision-one",
    )
    second = build_model_artifact_identity(
        tmp_path,
        requested_model_id="local-test",
        requested_revision="revision-one",
    )
    assert first == second
    assert {item.relative_path for item in first.files} == {
        "config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
    }

    (tmp_path / "vocab.json").write_text(json.dumps({"tokenizer": "two"}))
    changed = build_model_artifact_identity(
        tmp_path,
        requested_model_id="local-test",
        requested_revision="revision-one",
    )
    assert changed.manifest_sha256 != first.manifest_sha256

    (tmp_path / "speech_tokenizer" / "config.json").write_text(
        json.dumps({"model_type": "changed-codec"})
    )
    codec_changed = build_model_artifact_identity(
        tmp_path,
        requested_model_id="local-test",
        requested_revision="revision-one",
    )
    assert codec_changed.manifest_sha256 != changed.manifest_sha256


@pytest.mark.parametrize(
    "missing",
    [
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "speech_tokenizer/config.json",
        "speech_tokenizer/preprocessor_config.json",
        "speech_tokenizer/model.safetensors",
    ],
)
def test_source_manifest_rejects_incomplete_model_identity(tmp_path, missing: str) -> None:
    files = {
        "config.json": "{}",
        "tokenizer_config.json": "{}",
        "vocab.json": "{}",
        "merges.txt": "merge",
    }
    for name, value in files.items():
        if name != missing:
            (tmp_path / name).write_text(value)
    if missing != "model.safetensors":
        (tmp_path / "model.safetensors").write_bytes(b"weights")
    codec = tmp_path / "speech_tokenizer"
    codec.mkdir()
    if missing != "speech_tokenizer/config.json":
        (codec / "config.json").write_text("{}")
    if missing != "speech_tokenizer/preprocessor_config.json":
        (codec / "preprocessor_config.json").write_text("{}")
    if missing != "speech_tokenizer/model.safetensors":
        (codec / "model.safetensors").write_bytes(b"codec")
    with pytest.raises(RuntimeError, match="incomplete|missing"):
        build_model_artifact_identity(
            tmp_path,
            requested_model_id="local-test",
            requested_revision=None,
        )


def test_source_manifest_accepts_and_validates_sharded_checkpoint_and_tokenizer_json(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"talker.weight": "model-00001-of-00001.safetensors"}})
    )
    _write_codec_source(tmp_path)
    manifest = build_model_artifact_identity(
        tmp_path,
        requested_model_id="local-sharded",
        requested_revision=None,
    )
    assert len(manifest.files) == 8

    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(RuntimeError, match="missing indexed weight shards"):
        build_model_artifact_identity(
            tmp_path,
            requested_model_id="local-sharded",
            requested_revision=None,
        )


@pytest.mark.parametrize(
    "index",
    [
        "not-json",
        json.dumps({}),
        json.dumps({"weight_map": {}}),
        json.dumps({"weight_map": {"talker.weight": 3}}),
    ],
)
def test_source_manifest_rejects_malformed_or_empty_weight_index(tmp_path, index: str) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(index)
    _write_codec_source(tmp_path)

    with pytest.raises(RuntimeError, match="index|weights|shard"):
        build_model_artifact_identity(
            tmp_path,
            requested_model_id="local-malformed-index",
            requested_revision=None,
        )


@pytest.mark.parametrize("outside_kind", ["relative", "absolute"])
def test_source_manifest_rejects_indexed_shards_outside_snapshot(tmp_path, outside_kind: str) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (root / "config.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    shard = "../outside.safetensors" if outside_kind == "relative" else str(outside)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"talker.weight": shard}})
    )
    _write_codec_source(root)

    with pytest.raises(RuntimeError, match="outside|path|snapshot"):
        build_model_artifact_identity(
            root,
            requested_model_id="local-escape",
            requested_revision=None,
        )


def test_source_manifest_validates_present_index_even_with_single_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"single")
    (tmp_path / "model.safetensors.index.json").write_text("not-json")
    _write_codec_source(tmp_path)

    with pytest.raises(RuntimeError, match="index"):
        build_model_artifact_identity(
            tmp_path,
            requested_model_id="ambiguous-source",
            requested_revision=None,
        )


def test_default_remote_resolution_is_pinned_to_validated_snapshot(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolved = resolve_model_directory(ModelAssetConfig(local_files_only=True))

    assert resolved == tmp_path.resolve()
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "revision": VALIDATED_MODEL_REVISION,
            "cache_dir": None,
            "local_files_only": True,
        }
    ]


def test_nondefault_remote_resolution_requires_explicit_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: pytest.fail("unpinned source must be rejected before download"),
    )
    with pytest.raises(ValueError, match="revision"):
        resolve_model_directory(ModelAssetConfig(model_id="owner/unpinned-model"))

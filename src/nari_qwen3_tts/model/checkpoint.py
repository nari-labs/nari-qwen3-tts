"""Qwen3-TTS checkpoint identity, validation, and raw module loading."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from nari_qwen3_tts.config import ModelAssetConfig
from nari_qwen3_tts.contract.model import (
    FileDigest,
    ModelArtifactIdentity,
    ModelIdentityPolicy,
)
from nari_qwen3_tts.model.components import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from nari_qwen3_tts.model.config import Qwen3TTSConfig
from nari_qwen3_tts.model.loader import iter_safetensors_shards, load_hf_weights, require_all_parameters

_MANIFEST_NAMES = {
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
_REQUIRED_ROOT_FILES = {
    "config.json",
    "tokenizer_config.json",
}
_REQUIRED_CODEC_FILES = {
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
}


def _configure_model_math() -> None:
    """Apply the validated FP32 matmul policy before model execution."""

    torch.set_float32_matmul_precision("high")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolved_revision(path: Path) -> str | None:
    resolved = path.resolve()
    parents = list(resolved.parents)
    if resolved.parent.name == "snapshots":
        return resolved.name
    for parent in parents:
        if parent.name == "snapshots":
            relative = resolved.relative_to(parent)
            return relative.parts[0] if relative.parts else None
    return None


def _manifest_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name in _MANIFEST_NAMES:
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _validate_model_source(root: Path) -> None:
    missing = sorted(name for name in _REQUIRED_ROOT_FILES if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"Qwen3-TTS model source is incomplete; missing root files: {missing}")
    has_tokenizer_json = (root / "tokenizer.json").is_file()
    has_bpe_pair = (root / "vocab.json").is_file() and (root / "merges.txt").is_file()
    if not has_tokenizer_json and not has_bpe_pair:
        raise RuntimeError(
            "Qwen3-TTS model source is missing tokenizer.json or the vocab.json/merges.txt pair"
        )
    codec_root = root / "speech_tokenizer"
    missing_codec = sorted(name for name in _REQUIRED_CODEC_FILES if not (codec_root / name).is_file())
    if missing_codec:
        raise RuntimeError(
            "Qwen3-TTS model source is incomplete; missing speech_tokenizer files: "
            f"{missing_codec}"
        )
    single_path = root / "model.safetensors"
    index_path = root / "model.safetensors.index.json"
    if not single_path.is_file() and not index_path.is_file():
        raise RuntimeError(
            "Qwen3-TTS model source is missing model.safetensors or model.safetensors.index.json"
        )
    if not index_path.is_file():
        return
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"invalid Qwen3-TTS weight index: {index_path}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Qwen3-TTS weight index has no weights: {index_path}")
    shard_names = set(weight_map.values())
    if any(not isinstance(name, str) or not name.endswith(".safetensors") for name in shard_names):
        raise RuntimeError(f"Qwen3-TTS weight index contains an invalid shard path: {index_path}")
    resolved_root = root.resolve()
    shard_paths: dict[str, Path] = {}
    for name in shard_names:
        path = Path(name)
        resolved_path = (root / path).resolve()
        if path.is_absolute() or not resolved_path.is_relative_to(resolved_root):
            raise RuntimeError(
                f"Qwen3-TTS weight index shard path is outside the model snapshot: {name!r}"
            )
        shard_paths[name] = resolved_path
    missing_shards = sorted(name for name, path in shard_paths.items() if not path.is_file())
    if missing_shards:
        raise RuntimeError(f"Qwen3-TTS model source is missing indexed weight shards: {missing_shards}")


def build_model_artifact_identity(
    root: str | Path,
    *,
    requested_model_id: str,
    requested_revision: str | None,
) -> ModelArtifactIdentity:
    directory = Path(root).resolve()
    _validate_model_source(directory)
    files = tuple(
        FileDigest(
            relative_path=path.relative_to(directory).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in _manifest_paths(directory)
    )
    if not files:
        raise RuntimeError(f"model source manifest is empty: {directory}")
    canonical = json.dumps(
        [(item.relative_path, item.size_bytes, item.sha256) for item in files],
        separators=(",", ":"),
    ).encode()
    return ModelArtifactIdentity(
        requested_model_id=requested_model_id,
        requested_revision=requested_revision,
        resolved_directory=str(directory),
        resolved_revision=_resolved_revision(directory),
        files=files,
        manifest_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def resolve_model_directory(config: ModelAssetConfig) -> Path:
    candidate = Path(config.model_id)
    if candidate.is_dir():
        return candidate.resolve()
    revision = config.effective_revision
    if revision is None:
        raise ValueError("non-default remote Qwen3-TTS models require an explicit revision")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=config.model_id,
            revision=revision,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
    ).resolve()


@dataclass(slots=True)
class LoadedModelAssets:
    config: ModelAssetConfig
    identity: ModelArtifactIdentity
    model_config: Qwen3TTSConfig
    device: torch.device
    talker: Qwen3TTSTalkerModel
    code_predictor: Qwen3TTSCodePredictor
    code_predictor_layer0_embedding: nn.Embedding
    codec: nn.Module


class CheckpointLoader:
    """Validate one checkpoint and materialize its exact raw modules."""

    sample_rate = 24_000
    samples_per_frame = 1_920

    def __init__(
        self,
        config: ModelAssetConfig,
        *,
        identity_policy: ModelIdentityPolicy | None = None,
    ) -> None:
        self.asset_config = config
        self.local_directory = resolve_model_directory(config)
        self.identity = build_model_artifact_identity(
            self.local_directory,
            requested_model_id=config.model_id,
            requested_revision=config.effective_revision,
        )
        policy = identity_policy or ModelIdentityPolicy(
            expected_revision=(None if config.is_local_directory else config.effective_revision)
        )
        policy.validate(self.identity)
        if not config.is_local_directory and self.identity.resolved_revision is None:
            raise RuntimeError(
                "remote Qwen3-TTS resolution did not produce a pinned Hugging Face snapshot revision"
            )
        self.config = Qwen3TTSConfig.from_pretrained_dir(self.local_directory)

    @staticmethod
    def _materialize(module: nn.Module, *, device: torch.device) -> nn.Module:
        module.to_empty(device=device)
        return module

    def load_talker(self, *, device: torch.device, dtype: torch.dtype) -> Qwen3TTSTalkerModel:
        with torch.device("meta"):
            talker = Qwen3TTSTalkerModel(self.config).to(dtype=dtype)
        self._materialize(talker, device=device)
        prefix = "talker."
        weights = (
            (name.removeprefix(prefix), tensor)
            for name, tensor in iter_safetensors_shards(
                self.local_directory,
                device=device,
                prefix=prefix,
            )
            if not name.startswith("talker.code_predictor.")
        )
        loaded = load_hf_weights(talker, weights)
        require_all_parameters(talker, loaded, "Talker")
        return talker.eval()

    def load_code_predictor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Qwen3TTSCodePredictor, nn.Embedding]:
        with torch.device("meta"):
            predictor = Qwen3TTSCodePredictor(self.config).to(dtype=dtype)
            layer0 = nn.Embedding(self.config.talker.vocab_size, self.config.talker.hidden_size).to(dtype=dtype)
        self._materialize(predictor, device=device)
        self._materialize(layer0, device=device)
        prefix = "talker.code_predictor."
        weights = (
            (name.removeprefix(prefix), tensor)
            for name, tensor in iter_safetensors_shards(
                self.local_directory,
                device=device,
                prefix=prefix,
            )
        )
        loaded = load_hf_weights(predictor, weights)
        require_all_parameters(predictor, loaded, "Code Predictor")
        for _name, tensor in iter_safetensors_shards(
            self.local_directory,
            device=device,
            prefix="talker.model.codec_embedding.weight",
        ):
            layer0.weight.data.copy_(tensor)
            break
        else:
            raise RuntimeError("missing Talker layer-0 codec embedding")
        predictor.consolidate_stacked_weights()
        return predictor.eval(), layer0.eval()

    def load_codec(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        try:
            # Suppress qwen_tts's import-time flash-attn advisory banner.
            with contextlib.redirect_stdout(io.StringIO()):
                from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        except ImportError as error:
            raise RuntimeError("Codec loading requires the 'codec' package extra") from error
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(self.local_directory / "speech_tokenizer"),
            device_map=str(device),
            dtype=dtype,
        )
        return tokenizer.model.eval()

    def load_assets(self) -> LoadedModelAssets:
        _configure_model_math()
        device = torch.device(self.asset_config.device)
        _validate_device(self.asset_config, device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = torch.bfloat16
        talker = self.load_talker(device=device, dtype=dtype)
        predictor, layer0 = self.load_code_predictor(device=device, dtype=dtype)
        codec = self.load_codec(device=device, dtype=dtype).float()
        return LoadedModelAssets(
            config=self.asset_config,
            identity=self.identity,
            model_config=self.config,
            device=device,
            talker=talker,
            code_predictor=predictor,
            code_predictor_layer0_embedding=layer0,
            codec=codec,
        )


def _validate_device(config: ModelAssetConfig, device: torch.device) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA model loading requested but CUDA is unavailable")
        if config.require_h100 and "H100" not in torch.cuda.get_device_name(device):
            raise RuntimeError(f"production model loading requires H100, got {torch.cuda.get_device_name(device)!r}")
    elif config.require_h100:
        raise RuntimeError("require_h100=True requires a CUDA device")


def load_model_assets(
    config: ModelAssetConfig | None = None,
    *,
    identity_policy: ModelIdentityPolicy | None = None,
) -> LoadedModelAssets:
    asset_config = config or ModelAssetConfig()
    checkpoint = CheckpointLoader(asset_config, identity_policy=identity_policy)
    return checkpoint.load_assets()


__all__ = [
    "CheckpointLoader",
    "LoadedModelAssets",
    "build_model_artifact_identity",
    "load_model_assets",
    "resolve_model_directory",
]

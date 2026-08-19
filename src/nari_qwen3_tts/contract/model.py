"""Canonical immutable synthesis-model specification."""

from __future__ import annotations

from dataclasses import dataclass


def _require_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must contain hexadecimal characters") from error


@dataclass(frozen=True, slots=True)
class FileDigest:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("artifact file path must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("artifact file size must be an integer")
        if self.size_bytes < 0:
            raise ValueError("artifact file size must be non-negative")
        _require_sha256(self.sha256, label="artifact file SHA256")


@dataclass(frozen=True, slots=True)
class ModelArtifactIdentity:
    """Content identity of one resolved Hugging Face snapshot."""

    requested_model_id: str
    requested_revision: str | None
    resolved_directory: str
    resolved_revision: str | None
    files: tuple[FileDigest, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.requested_model_id, str) or not self.requested_model_id:
            raise ValueError("requested model ID must be a non-empty string")
        if not isinstance(self.resolved_directory, str) or not self.resolved_directory:
            raise ValueError("resolved model directory must be a non-empty string")
        for label, value in (
            ("requested revision", self.requested_revision),
            ("resolved revision", self.resolved_revision),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{label} must be a non-empty string when provided")
        if not isinstance(self.files, tuple) or not self.files:
            raise ValueError("model artifact identity requires file digests")
        if any(not isinstance(item, FileDigest) for item in self.files):
            raise TypeError("model artifact files must contain FileDigest values")
        _require_sha256(self.manifest_sha256, label="model manifest SHA256")


@dataclass(frozen=True, slots=True)
class ModelIdentityPolicy:
    """Optional fail-closed expectations for one resolved model artifact."""

    expected_revision: str | None = None
    expected_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.expected_revision is not None and (
            not isinstance(self.expected_revision, str) or not self.expected_revision
        ):
            raise ValueError("expected model revision must be a non-empty string")
        if self.expected_manifest_sha256 is not None:
            _require_sha256(
                self.expected_manifest_sha256,
                label="expected model manifest SHA256",
            )

    def validate(self, identity: ModelArtifactIdentity) -> None:
        if not isinstance(identity, ModelArtifactIdentity):
            raise TypeError("model identity policy requires a ModelArtifactIdentity")
        if (
            self.expected_revision is not None
            and identity.resolved_revision != self.expected_revision
        ):
            raise RuntimeError(
                "resolved model revision does not match the configured identity policy"
            )
        if (
            self.expected_manifest_sha256 is not None
            and identity.manifest_sha256 != self.expected_manifest_sha256
        ):
            raise RuntimeError(
                "resolved model manifest does not match the configured identity policy"
            )


@dataclass(frozen=True, slots=True)
class SynthesisModelSpec:
    codec_eos_token_id: int
    talker_vocab_size: int
    text_vocab_size: int = 151_936
    num_codebooks: int = 16
    samples_per_frame: int = 1_920
    sample_rate: int = 24_000

    def __post_init__(self) -> None:
        if self.num_codebooks != 16:
            raise ValueError("Qwen3-TTS synthesis requires exactly 16 codebooks")
        if not 0 <= self.codec_eos_token_id < self.talker_vocab_size:
            raise ValueError("Codec EOS token must fit the Talker vocabulary")
        if self.text_vocab_size < 1:
            raise ValueError("text_vocab_size must be positive")
        if self.samples_per_frame < 1:
            raise ValueError("samples_per_frame must be positive")
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")


__all__ = [
    "FileDigest",
    "ModelArtifactIdentity",
    "ModelIdentityPolicy",
    "SynthesisModelSpec",
]

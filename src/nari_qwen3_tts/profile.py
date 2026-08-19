"""Validated process-start configuration for fixed Qwen3-TTS CUDA execution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} setting(s): {', '.join(unknown)}")


def _require_keys(value: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Missing {name} setting(s): {', '.join(missing)}")


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _integer_tuple(name: str, value: object, *, unique: bool = True) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    _sizes(name, result, unique=unique)
    return result


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _sizes(name: str, values: tuple[int, ...], *, unique: bool = True) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    for value in values:
        _positive(f"{name} entry", value)
    if unique and any(left >= right for left, right in zip(values, values[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")


class ExecutionProfile(str, Enum):
    TTFA = "ttfa"
    BALANCED = "balanced"
    THROUGHPUT = "throughput"


class RequiredSchedulingPolicy(str, Enum):
    """Scheduling policy required by a deployment profile."""

    ROUND_ROBIN = "round_robin"
    DEADLINE_AWARE = "deadline_aware"


@dataclass(frozen=True, slots=True)
class BatchCaptureConfig:
    max_batch_size: int
    batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        _positive("max_batch_size", self.max_batch_size)
        _sizes("batch_sizes", self.batch_sizes)
        if self.batch_sizes[-1] > self.max_batch_size:
            raise ValueError("capture batch size exceeds max_batch_size")


@dataclass(frozen=True, slots=True)
class PrefillCaptureConfig(BatchCaptureConfig):
    token_buckets: tuple[int, ...]
    exact_sequence_lengths: tuple[int, ...]
    exact_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        super(PrefillCaptureConfig, self).__post_init__()
        _sizes("token_buckets", self.token_buckets)
        _sizes("exact_sequence_lengths", self.exact_sequence_lengths)
        _sizes("exact_batch_sizes", self.exact_batch_sizes)
        if self.exact_batch_sizes[-1] > self.max_batch_size:
            raise ValueError("exact prefill capture batch size exceeds max_batch_size")


@dataclass(frozen=True, slots=True)
class CodecBatchCaptureConfig:
    whole_sequence_first_frame: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    whole_sequence_followup: tuple[int, ...] = (1, 2, 4, 8)
    cold: tuple[int, ...] = (1, 2, 4, 8)
    cold_terminal_partial: tuple[int, ...] = ()
    warm_partial: tuple[int, ...] = tuple(range(1, 9))
    warm_full: tuple[int, ...] = (*range(1, 9), 16, 32)

    def __post_init__(self) -> None:
        for name in ("whole_sequence_first_frame", "whole_sequence_followup", "cold", "warm_partial", "warm_full"):
            _sizes(name, getattr(self, name))
        if self.cold_terminal_partial:
            _sizes("cold_terminal_partial", self.cold_terminal_partial)


@dataclass(frozen=True, slots=True)
class CodecFrameCaptureConfig:
    cold: tuple[int, ...] = tuple(range(4, 8))
    cold_terminal_partial: tuple[int, ...] = ()
    warm: tuple[int, ...] = tuple(range(1, 13))
    warm_full: tuple[int, ...] = (12,)
    terminal_pad: tuple[int, ...] = (12,)

    def __post_init__(self) -> None:
        for name in ("cold", "warm", "warm_full", "terminal_pad"):
            _sizes(name, getattr(self, name))
        if self.cold_terminal_partial:
            _sizes("cold_terminal_partial", self.cold_terminal_partial)
        if not set(self.cold_terminal_partial) <= set(self.cold):
            raise ValueError("cold_terminal_partial contains an uncaptured cold frame size")
        if not set(self.warm_full) <= set(self.warm):
            raise ValueError("warm_full contains an uncaptured warm frame size")
        if not set(self.terminal_pad) <= set(self.warm_full):
            raise ValueError("terminal_pad must use full-cohort warm captures")


@dataclass(frozen=True, slots=True)
class CodecCaptureConfig:
    max_batch_size: int
    chunk_schedule: tuple[int, ...]
    suppressed_bootstrap_chunk_schedule: tuple[int, ...]
    batches: CodecBatchCaptureConfig = field(default_factory=CodecBatchCaptureConfig)
    frames: CodecFrameCaptureConfig = field(default_factory=CodecFrameCaptureConfig)

    def __post_init__(self) -> None:
        _positive("Codec max_batch_size", self.max_batch_size)
        _sizes("chunk_schedule", self.chunk_schedule, unique=False)
        _sizes(
            "suppressed_bootstrap_chunk_schedule",
            self.suppressed_bootstrap_chunk_schedule,
            unique=False,
        )
        for name in (
            "whole_sequence_first_frame",
            "whole_sequence_followup",
            "cold",
            "cold_terminal_partial",
            "warm_partial",
            "warm_full",
        ):
            if not getattr(self.batches, name):
                continue
            if getattr(self.batches, name)[-1] > self.max_batch_size:
                raise ValueError(f"Codec {name} capture exceeds max_batch_size")


@dataclass(frozen=True, slots=True)
class _ProfileDocument:
    """Private merged representation used while resolving one profile."""

    profile: ExecutionProfile
    required_policy: RequiredSchedulingPolicy
    pressing_lead_s: float | None
    talker_prefill: PrefillCaptureConfig
    talker_decode: BatchCaptureConfig
    code_predictor: BatchCaptureConfig
    codec: CodecCaptureConfig
    talker_capture_slots: int = 4
    # 256 x 128-token pages cover one full-length request or many normal TTS
    # requests. Exhaustion raises KVAllocationError rather than degrading, so
    # raising this capacity is a deliberate memory trade.
    kv_pages: int = 256
    kv_page_size: int = 128
    workspace_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ExecutionProfile):
            raise TypeError("profile must be an ExecutionProfile")
        if not isinstance(self.required_policy, RequiredSchedulingPolicy):
            raise TypeError("required_policy must be a RequiredSchedulingPolicy")
        if self.pressing_lead_s is not None and (
            isinstance(self.pressing_lead_s, bool)
            or not isinstance(self.pressing_lead_s, (int, float))
            or not math.isfinite(self.pressing_lead_s)
        ):
            raise TypeError("pressing_lead_s must be a finite numeric value or None")
        for name in ("talker_capture_slots", "kv_pages", "kv_page_size", "workspace_bytes"):
            _positive(name, getattr(self, name))
        if self.required_policy is RequiredSchedulingPolicy.DEADLINE_AWARE:
            if self.pressing_lead_s is None or self.pressing_lead_s <= 0:
                raise ValueError("deadline-aware scheduling requires a positive lead_s")
        elif self.pressing_lead_s is not None:
            raise ValueError("pressing_lead_s is valid only for deadline-aware scheduling")

    @classmethod
    def for_profile(cls, profile: ExecutionProfile | str) -> _ProfileDocument:
        profile = ExecutionProfile(profile)
        resource = files("nari_qwen3_tts.profiles").joinpath(f"{profile.value}.yaml")
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        mapping = _mapping(raw, f"packaged {profile.value} profile")
        config = cls._from_complete_mapping(mapping)
        if config.profile is not profile:
            raise RuntimeError(
                f"packaged {profile.value} profile declares {config.profile.value}"
            )
        return config

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any] | _ProfileDocument,
        *,
        base: _ProfileDocument | ExecutionProfile | str | None = None,
    ) -> _ProfileDocument:
        if isinstance(value, cls):
            return value
        raw = _mapping(value, "profile")
        allowed = {
            "extends",
            "profile",
            "required_policy",
            "pressing_lead_s",
            "talker_prefill",
            "talker_decode",
            "code_predictor",
            "codec",
            "talker_capture_slots",
            "kv_pages",
            "kv_page_size",
            "workspace_bytes",
        }
        _reject_unknown(raw, allowed, "profile")

        declared_profile = raw.get("extends", raw.get("profile"))
        if base is None:
            if declared_profile is None:
                raise ValueError("profile overlay requires extends/profile or an explicit base")
            resolved_base = cls.for_profile(ExecutionProfile(declared_profile))
        elif isinstance(base, cls):
            resolved_base = base
        else:
            resolved_base = cls.for_profile(ExecutionProfile(base))
        if declared_profile is not None and ExecutionProfile(declared_profile) is not resolved_base.profile:
            raise ValueError("profile overlay profile does not match its base")
        if "extends" in raw and "profile" in raw:
            if ExecutionProfile(raw["extends"]) is not ExecutionProfile(raw["profile"]):
                raise ValueError("profile extends and profile must agree")

        return cls(
            profile=resolved_base.profile,
            required_policy=RequiredSchedulingPolicy(
                raw.get("required_policy", resolved_base.required_policy)
            ),
            pressing_lead_s=raw.get("pressing_lead_s", resolved_base.pressing_lead_s),
            talker_prefill=cls._prefill_section(
                raw.get("talker_prefill"), resolved_base.talker_prefill
            ),
            talker_decode=cls._batch_section(
                raw.get("talker_decode"), resolved_base.talker_decode, "profile.talker_decode"
            ),
            code_predictor=cls._batch_section(
                raw.get("code_predictor"),
                resolved_base.code_predictor,
                "profile.code_predictor",
            ),
            codec=cls._codec_section(raw.get("codec"), resolved_base.codec),
            talker_capture_slots=raw.get(
                "talker_capture_slots", resolved_base.talker_capture_slots
            ),
            kv_pages=raw.get("kv_pages", resolved_base.kv_pages),
            kv_page_size=raw.get("kv_page_size", resolved_base.kv_page_size),
            workspace_bytes=raw.get("workspace_bytes", resolved_base.workspace_bytes),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        base: _ProfileDocument | ExecutionProfile | str | None = None,
    ) -> _ProfileDocument:
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        return cls.from_dict(_mapping(raw, f"profile YAML {source}"), base=base)

    @staticmethod
    def _batch_section(
        value: object,
        default: BatchCaptureConfig,
        name: str,
    ) -> BatchCaptureConfig:
        if value is None:
            return default
        raw = _mapping(value, name)
        _reject_unknown(raw, {"max_batch_size", "batch_sizes"}, name)
        return BatchCaptureConfig(
            max_batch_size=raw.get("max_batch_size", default.max_batch_size),
            batch_sizes=(
                _integer_tuple(f"{name}.batch_sizes", raw["batch_sizes"])
                if "batch_sizes" in raw
                else default.batch_sizes
            ),
        )

    @staticmethod
    def _prefill_section(value: object, default: PrefillCaptureConfig) -> PrefillCaptureConfig:
        if value is None:
            return default
        name = "profile.talker_prefill"
        raw = _mapping(value, name)
        _reject_unknown(
            raw,
            {
                "max_batch_size",
                "batch_sizes",
                "token_buckets",
                "exact_sequence_lengths",
                "exact_batch_sizes",
            },
            name,
        )

        def values(field_name: str, default_value: tuple[int, ...]) -> tuple[int, ...]:
            if field_name not in raw:
                return default_value
            return _integer_tuple(f"{name}.{field_name}", raw[field_name])

        return PrefillCaptureConfig(
            max_batch_size=raw.get("max_batch_size", default.max_batch_size),
            batch_sizes=values("batch_sizes", default.batch_sizes),
            token_buckets=values("token_buckets", default.token_buckets),
            exact_sequence_lengths=values(
                "exact_sequence_lengths", default.exact_sequence_lengths
            ),
            exact_batch_sizes=values("exact_batch_sizes", default.exact_batch_sizes),
        )

    @staticmethod
    def _codec_section(value: object, default: CodecCaptureConfig) -> CodecCaptureConfig:
        if value is None:
            return default
        name = "profile.codec"
        raw = _mapping(value, name)
        _reject_unknown(
            raw,
            {
                "max_batch_size",
                "chunk_schedule",
                "suppressed_bootstrap_chunk_schedule",
                "batches",
                "frames",
            },
            name,
        )
        batch_raw = _mapping(raw.get("batches", {}), f"{name}.batches")
        batch_fields = {
            "whole_sequence_first_frame",
            "whole_sequence_followup",
            "cold",
            "cold_terminal_partial",
            "warm_partial",
            "warm_full",
        }
        _reject_unknown(batch_raw, batch_fields, f"{name}.batches")

        def batch_values(field_name: str) -> tuple[int, ...]:
            if field_name not in batch_raw:
                return getattr(default.batches, field_name)
            value = batch_raw[field_name]
            if field_name == "cold_terminal_partial" and not value:
                return ()
            return _integer_tuple(f"{name}.batches.{field_name}", value)

        frame_raw = _mapping(raw.get("frames", {}), f"{name}.frames")
        frame_fields = {"cold", "cold_terminal_partial", "warm", "warm_full", "terminal_pad"}
        _reject_unknown(frame_raw, frame_fields, f"{name}.frames")

        def frame_values(field_name: str) -> tuple[int, ...]:
            if field_name not in frame_raw:
                return getattr(default.frames, field_name)
            value = frame_raw[field_name]
            if field_name == "cold_terminal_partial" and not value:
                return ()
            return _integer_tuple(f"{name}.frames.{field_name}", value)

        return CodecCaptureConfig(
            max_batch_size=raw.get("max_batch_size", default.max_batch_size),
            chunk_schedule=(
                _integer_tuple(
                    f"{name}.chunk_schedule", raw["chunk_schedule"], unique=False
                )
                if "chunk_schedule" in raw
                else default.chunk_schedule
            ),
            suppressed_bootstrap_chunk_schedule=(
                _integer_tuple(
                    f"{name}.suppressed_bootstrap_chunk_schedule",
                    raw["suppressed_bootstrap_chunk_schedule"],
                    unique=False,
                )
                if "suppressed_bootstrap_chunk_schedule" in raw
                else default.suppressed_bootstrap_chunk_schedule
            ),
            batches=CodecBatchCaptureConfig(
                **{field_name: batch_values(field_name) for field_name in batch_fields}
            ),
            frames=CodecFrameCaptureConfig(
                **{field_name: frame_values(field_name) for field_name in frame_fields}
            ),
        )

    @classmethod
    def _from_complete_mapping(cls, value: Mapping[str, Any]) -> _ProfileDocument:
        required = {
            "profile",
            "required_policy",
            "pressing_lead_s",
            "talker_prefill",
            "talker_decode",
            "code_predictor",
            "codec",
            "talker_capture_slots",
            "kv_pages",
            "kv_page_size",
            "workspace_bytes",
        }
        _reject_unknown(value, required, "packaged profile")
        _require_keys(value, required, "packaged profile")
        nested_required = {
            "talker_prefill": {
                "max_batch_size",
                "batch_sizes",
                "token_buckets",
                "exact_sequence_lengths",
                "exact_batch_sizes",
            },
            "talker_decode": {"max_batch_size", "batch_sizes"},
            "code_predictor": {"max_batch_size", "batch_sizes"},
            "codec": {
                "max_batch_size",
                "chunk_schedule",
                "suppressed_bootstrap_chunk_schedule",
                "batches",
                "frames",
            },
        }
        for section, fields in nested_required.items():
            _require_keys(
                _mapping(value[section], f"packaged profile.{section}"),
                fields,
                f"packaged profile.{section}",
            )
        _require_keys(
            _mapping(value["codec"]["batches"], "packaged profile.codec.batches"),
            {
                "whole_sequence_first_frame",
                "whole_sequence_followup",
                "cold",
                "cold_terminal_partial",
                "warm_partial",
                "warm_full",
            },
            "packaged profile.codec.batches",
        )
        _require_keys(
            _mapping(value["codec"]["frames"], "packaged profile.codec.frames"),
            {"cold", "cold_terminal_partial", "warm", "warm_full", "terminal_pad"},
            "packaged profile.codec.frames",
        )
        # Complete profile files still pass through the exact same strict section parsers as overlays.
        seed = cls(
            profile=ExecutionProfile(value["profile"]),
            required_policy=RequiredSchedulingPolicy(value["required_policy"]),
            pressing_lead_s=value["pressing_lead_s"],
            talker_prefill=PrefillCaptureConfig(1, (1,), (1,), (1,), (1,)),
            talker_decode=BatchCaptureConfig(1, (1,)),
            code_predictor=BatchCaptureConfig(1, (1,)),
            codec=CodecCaptureConfig(
                1,
                (1,),
                (1,),
                CodecBatchCaptureConfig(
                    whole_sequence_first_frame=(1,),
                    whole_sequence_followup=(1,),
                    cold=(1,),
                    warm_partial=(1,),
                    warm_full=(1,),
                ),
                CodecFrameCaptureConfig(
                    cold=(1,),
                    warm=(1,),
                    warm_full=(1,),
                    terminal_pad=(1,),
                ),
            ),
        )
        return cls(
            profile=ExecutionProfile(value["profile"]),
            required_policy=RequiredSchedulingPolicy(value["required_policy"]),
            pressing_lead_s=value["pressing_lead_s"],
            talker_prefill=cls._prefill_section(value["talker_prefill"], seed.talker_prefill),
            talker_decode=cls._batch_section(
                value["talker_decode"], seed.talker_decode, "profile.talker_decode"
            ),
            code_predictor=cls._batch_section(
                value["code_predictor"], seed.code_predictor, "profile.code_predictor"
            ),
            codec=cls._codec_section(value["codec"], seed.codec),
            talker_capture_slots=value["talker_capture_slots"],
            kv_pages=value["kv_pages"],
            kv_page_size=value["kv_page_size"],
            workspace_bytes=value["workspace_bytes"],
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["profile"] = self.profile.value
        value["required_policy"] = self.required_policy.value
        return value

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def codec_chunks(self, *, silent_bootstrap_suppressed: bool) -> tuple[int, ...]:
        return (
            self.codec.suppressed_bootstrap_chunk_schedule
            if silent_bootstrap_suppressed
            else self.codec.chunk_schedule
        )


@dataclass(frozen=True, slots=True)
class SchedulingPolicyConfig:
    """Scheduling policy settings consumed only by the planner."""

    kind: RequiredSchedulingPolicy
    pressing_lead_s: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RequiredSchedulingPolicy):
            raise TypeError("scheduling policy kind must be a RequiredSchedulingPolicy")
        if self.kind is RequiredSchedulingPolicy.DEADLINE_AWARE:
            if self.pressing_lead_s is None:
                raise ValueError("deadline-aware scheduling requires a lead")
            if (
                isinstance(self.pressing_lead_s, bool)
                or not isinstance(self.pressing_lead_s, (int, float))
                or not math.isfinite(self.pressing_lead_s)
                or self.pressing_lead_s < 0
            ):
                raise ValueError("deadline-aware scheduling lead must be finite and non-negative")
        elif self.pressing_lead_s is not None:
            raise ValueError("round-robin scheduling cannot carry a pressing lead")


@dataclass(frozen=True, slots=True)
class StageExecutionConfig:
    """Canonical definition of every stage batch limit and CUDA capture shape."""

    talker_prefill: PrefillCaptureConfig
    talker_decode: BatchCaptureConfig
    code_predictor: BatchCaptureConfig
    codec: CodecCaptureConfig


@dataclass(frozen=True, slots=True)
class ExecutorResourceConfig:
    """Fixed CUDA resource capacities consumed only by execution."""

    talker_capture_slots: int
    kv_pages: int
    kv_page_size: int
    workspace_bytes: int


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """Immutable startup result shared by composition, planner, and executor."""

    name: str
    sha256: str
    policy: SchedulingPolicyConfig
    stages: StageExecutionConfig
    resources: ExecutorResourceConfig
    _canonical_json: str = field(repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResolvedProfile:
        document = _ProfileDocument._from_complete_mapping(value)
        canonical = document.canonical_json()
        return cls(
            name=document.profile.value,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            policy=SchedulingPolicyConfig(
                document.required_policy,
                document.pressing_lead_s,
            ),
            stages=StageExecutionConfig(
                talker_prefill=document.talker_prefill,
                talker_decode=document.talker_decode,
                code_predictor=document.code_predictor,
                codec=document.codec,
            ),
            resources=ExecutorResourceConfig(
                talker_capture_slots=document.talker_capture_slots,
                kv_pages=document.kv_pages,
                kv_page_size=document.kv_page_size,
                workspace_bytes=document.workspace_bytes,
            ),
            _canonical_json=canonical,
        )

    def canonical_json(self) -> str:
        return self._canonical_json

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    def codec_chunks(self, *, silent_bootstrap_suppressed: bool) -> tuple[int, ...]:
        return (
            self.stages.codec.suppressed_bootstrap_chunk_schedule
            if silent_bootstrap_suppressed
            else self.stages.codec.chunk_schedule
        )


class ProfileLoader:
    """Resolve packaged profiles and YAML overlays before model loading begins."""

    def load_profile(self, profile: ExecutionProfile | str) -> ResolvedProfile:
        document = _ProfileDocument.for_profile(profile)
        return ResolvedProfile.from_mapping(document.to_dict())

    def load_mapping(
        self,
        value: Mapping[str, Any],
        *,
        base: ExecutionProfile | ResolvedProfile | str | None = None,
    ) -> ResolvedProfile:
        document_base: _ProfileDocument | ExecutionProfile | str | None = base
        if isinstance(base, ResolvedProfile):
            document_base = _ProfileDocument._from_complete_mapping(base.to_dict())
        document = _ProfileDocument.from_dict(value, base=document_base)
        return ResolvedProfile.from_mapping(document.to_dict())

    def load_yaml(
        self,
        path: str | Path,
        *,
        base: ExecutionProfile | ResolvedProfile | str | None = None,
    ) -> ResolvedProfile:
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        return self.load_mapping(
            _mapping(raw, f"profile YAML {source}"),
            base=base,
        )


__all__ = [
    "BatchCaptureConfig",
    "CodecBatchCaptureConfig",
    "CodecCaptureConfig",
    "CodecFrameCaptureConfig",
    "ExecutionProfile",
    "ExecutorResourceConfig",
    "PrefillCaptureConfig",
    "ProfileLoader",
    "RequiredSchedulingPolicy",
    "ResolvedProfile",
    "SchedulingPolicyConfig",
    "StageExecutionConfig",
]

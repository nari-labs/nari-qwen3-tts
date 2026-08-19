"""Engine readiness and CUDA execution health values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nari_qwen3_tts.contract.stage import SynthesisStage

_STAGE_VALUES = frozenset(stage.value for stage in SynthesisStage)


def _non_negative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} count must be an integer")
    if value < 0:
        raise ValueError(f"{label} count must be non-negative")
    return value


class ServicePhase(str, Enum):
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class StageStats:
    name: SynthesisStage
    submitted: int
    replayed: int
    failed: int

    def __post_init__(self) -> None:
        if isinstance(self.name, str):
            try:
                object.__setattr__(self, "name", SynthesisStage(self.name))
            except ValueError as error:
                raise ValueError(
                    "stage statistics must identify one synthesis stage"
                ) from error
        elif not isinstance(self.name, SynthesisStage):
            raise TypeError("stage statistics name must identify one synthesis stage")
        for name in ("submitted", "replayed", "failed"):
            _non_negative_integer(getattr(self, name), label=name)
        if self.replayed + self.failed > self.submitted:
            raise ValueError("stage accounting cannot exceed submitted work")

    @property
    def accounted(self) -> bool:
        return self.submitted == self.replayed + self.failed


@dataclass(frozen=True, slots=True)
class EngineHealth:
    required_keys: int
    captured_keys: int
    required_cuda_graph_instances: int
    captured_cuda_graph_instances: int
    capture_failures: int
    eager_fallbacks: int
    stages: tuple[StageStats, ...]
    metadata_actions: int = 0

    def __post_init__(self) -> None:
        for name in (
            "required_keys",
            "captured_keys",
            "required_cuda_graph_instances",
            "captured_cuda_graph_instances",
            "capture_failures",
            "eager_fallbacks",
            "metadata_actions",
        ):
            _non_negative_integer(getattr(self, name), label=f"health {name}")
        if not isinstance(self.stages, tuple) or any(
            not isinstance(stage, StageStats) for stage in self.stages
        ):
            raise TypeError("health stages must be a tuple of StageStats values")
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("health stage statistic names cannot be duplicated")

    @property
    def capture_ready(self) -> bool:
        return (
            self.required_keys > 0
            and self.required_keys == self.captured_keys
            and self.required_cuda_graph_instances > 0
            and self.required_cuda_graph_instances == self.captured_cuda_graph_instances
            and self.capture_failures == 0
            and self.eager_fallbacks == 0
            and len(self.stages) == 4
            and {stage.name for stage in self.stages} == frozenset(SynthesisStage)
            and all(stage.accounted and stage.failed == 0 for stage in self.stages)
        )

    def submitted_by_stage(self) -> dict[str, int]:
        return {stage.name.value: stage.submitted for stage in self.stages}

    def stage(self, name: SynthesisStage) -> StageStats:
        if not isinstance(name, SynthesisStage):
            raise TypeError("stage lookup requires a SynthesisStage")
        return next(stage for stage in self.stages if stage.name is name)

    @property
    def ready(self) -> bool:
        """Execution-facing spelling for the same fail-closed capture verdict."""

        return self.capture_ready


@dataclass(frozen=True, slots=True)
class ReadinessStatus:
    phase: ServicePhase
    ready: bool
    reason: str
    ordinary_pcm_bytes: int
    ordinary_stage_deltas: tuple[tuple[str, int], ...]
    ordinary_retired: bool
    health: EngineHealth


def evaluate_readiness(
    *,
    phase: ServicePhase,
    health: EngineHealth,
    ordinary_pcm_bytes: int,
    ordinary_stage_deltas: dict[str, int],
    ordinary_retired: bool,
    failed_after_ready: bool = False,
) -> ReadinessStatus:
    """Explain readiness with the reason that actually applies."""

    pcm_valid = (
        not isinstance(ordinary_pcm_bytes, bool)
        and isinstance(ordinary_pcm_bytes, int)
        and ordinary_pcm_bytes > 0
        and ordinary_pcm_bytes % 2 == 0
    )
    stage_deltas_valid = set(ordinary_stage_deltas) == _STAGE_VALUES and all(
        not isinstance(ordinary_stage_deltas[name], bool)
        and isinstance(ordinary_stage_deltas[name], int)
        and ordinary_stage_deltas[name] > 0
        for name in _STAGE_VALUES
    )
    if not health.capture_ready:
        reason = "capture_health_invalid"
    elif not pcm_valid:
        reason = "ordinary_tts_not_proven"
    elif not stage_deltas_valid:
        reason = "ordinary_stage_coverage_incomplete"
    elif not ordinary_retired:
        reason = "ordinary_tts_not_retired"
    elif phase is ServicePhase.FAILED:
        reason = "service_failed" if failed_after_ready else "startup_failed"
    elif phase is ServicePhase.STOPPED:
        reason = "service_stopped"
    elif phase is not ServicePhase.READY:
        reason = "service_starting"
    else:
        reason = "ready"
    return ReadinessStatus(
        phase=phase,
        ready=reason == "ready",
        reason=reason,
        ordinary_pcm_bytes=ordinary_pcm_bytes,
        ordinary_stage_deltas=tuple(sorted(ordinary_stage_deltas.items())),
        ordinary_retired=ordinary_retired,
        health=health,
    )


__all__ = [
    "EngineHealth",
    "ReadinessStatus",
    "ServicePhase",
    "StageStats",
    "evaluate_readiness",
]

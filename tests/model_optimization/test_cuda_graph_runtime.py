from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from nari_qwen3_tts.contract import SynthesisStage
from nari_qwen3_tts.executor import (
    CaptureStartupError,
    UncapturedExecutionError,
)
from nari_qwen3_tts.executor.executor import Executor
from nari_qwen3_tts.executor.types import CodePredictorResult
from nari_qwen3_tts.planner import CaptureCatalog
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader


@dataclass
class _Stage:
    captured: set[object] = field(default_factory=set)
    replayed: list[tuple[object, object]] = field(default_factory=list)
    fail_on: object | None = None
    capture_slots: int = 1

    @property
    def captured_cuda_graph_instances(self) -> int:
        return len(self.captured) * self.capture_slots

    def capture(self, key: object) -> None:
        if key == self.fail_on:
            raise RuntimeError("capture failed")
        self.captured.add(key)

    def replay(self, key: object, values: object) -> object:
        if key not in self.captured:
            raise RuntimeError("not captured")
        self.replayed.append((key, values))
        return values


def _runtime() -> tuple[
    Executor,
    dict[SynthesisStage, _Stage],
    CaptureCatalog,
]:
    config = ProfileLoader().load_profile(ExecutionProfile.TTFA)
    catalog = CaptureCatalog.from_config(config.stages)
    stages = {name: _Stage() for name in SynthesisStage}
    stages[SynthesisStage.TALKER_PREFILL].capture_slots = config.resources.talker_capture_slots
    stages[SynthesisStage.TALKER_DECODE] = stages[SynthesisStage.TALKER_PREFILL]
    return (
        Executor(
            config=config,
            required_keys=catalog.required_keys,
            talker=stages[SynthesisStage.TALKER_PREFILL],
            code_predictor=stages[SynthesisStage.CODE_PREDICTOR],
            codec=stages[SynthesisStage.CODEC],
            optimizations=None,
        ),
        stages,
        catalog,
    )


def test_startup_captures_every_declared_key_and_graph_instance_before_ready() -> None:
    runtime, _, _ = _runtime()
    runtime.capture_all()
    health = runtime.health()
    assert health.ready
    assert health.required_keys == health.captured_keys
    assert health.required_cuda_graph_instances == health.captured_cuda_graph_instances
    assert health.capture_failures == 0
    assert health.eager_fallbacks == 0


def test_capture_failure_is_fail_closed_and_startup_is_one_shot() -> None:
    runtime, stages, catalog = _runtime()
    stages[SynthesisStage.CODE_PREDICTOR].fail_on = next(iter(catalog.code_predictor))
    with pytest.raises(CaptureStartupError):
        runtime.capture_all()
    assert not runtime.health().ready
    with pytest.raises(CaptureStartupError, match="only once"):
        runtime.capture_all()


def test_partial_startup_capture_never_leaves_an_executable_runtime() -> None:
    runtime, stages, catalog = _runtime()
    # Code Predictor is captured last, so Talker and Codec have already installed
    # captures when this failure is raised. Those partial captures must remain unusable.
    stages[SynthesisStage.CODE_PREDICTOR].fail_on = next(iter(catalog.code_predictor))
    with pytest.raises(CaptureStartupError):
        runtime.capture_all()

    talker_key = next(iter(catalog.talker_decode))
    with pytest.raises(UncapturedExecutionError, match="startup did not complete"):
        runtime.talker_decode_rows(talker_key, values="work")
    assert stages[SynthesisStage.TALKER_DECODE].replayed == []


def test_nonempty_submission_without_capture_is_invalid_not_eager() -> None:
    runtime, _, catalog = _runtime()
    key = next(iter(catalog.code_predictor))
    with pytest.raises(UncapturedExecutionError):
        runtime.code_predictor_rows(key, values="work")
    health = runtime.health().stage(SynthesisStage.CODE_PREDICTOR)
    assert health.submitted == 1
    assert health.replayed == 0
    assert health.failed == 1
    assert runtime.health().eager_fallbacks == 0


def test_typed_replay_accounting_and_empty_terminal_are_separate() -> None:
    runtime, _, catalog = _runtime()
    runtime.capture_all()
    key = next(iter(catalog.code_predictor))
    output = CodePredictorResult(
        frames=torch.zeros((1, 16), dtype=torch.long),
        codec_sum=torch.zeros((1, 4)),
    )
    assert runtime.code_predictor_rows(key, values=output) is output
    runtime.empty_terminal(rows=2)
    health = runtime.health()
    assert health.stage(SynthesisStage.CODE_PREDICTOR).replayed == 1
    assert health.metadata_actions == 1
    assert all(stage.accounted for stage in health.stages)


def test_runtime_rejects_a_captured_key_from_the_wrong_typed_stage() -> None:
    runtime, stages, catalog = _runtime()
    runtime.capture_all()
    talker_key = next(iter(catalog.talker_decode))

    with pytest.raises(TypeError, match="Code Predictor.*capture key"):
        runtime.code_predictor_rows(talker_key, values="work")  # type: ignore[arg-type]

    assert stages[SynthesisStage.CODE_PREDICTOR].replayed == []
    health = runtime.health().stage(SynthesisStage.CODE_PREDICTOR)
    assert health.submitted == 1
    assert health.replayed == 0
    assert health.failed == 1

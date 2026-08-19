from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


@dataclass
class _Runtime:
    result: object
    calls: list[tuple[str, object, object]]

    def replay_code_predictor(self, key, values):
        self.calls.append(("code_predictor", key, values))
        return self.result


def _predictor_decision(*, batch_id: int = 1, decision_id: int = 1):
    from nari_qwen3_tts.contract import (
        CodePredictorBatchCompatibility,
        CodePredictorCaptureKey,
        CudaGraphRef,
        ScheduleDecision,
        StageBatchRow,
        StageExecutionBatch,
        SynthesisStage,
    )

    compatibility = CodePredictorBatchCompatibility()
    batch = StageExecutionBatch(
        batch_id=batch_id,
        decision_id=decision_id,
        stage=SynthesisStage.CODE_PREDICTOR,
        compatibility=compatibility,
        capture=CudaGraphRef(
            SynthesisStage.CODE_PREDICTOR,
            CodePredictorCaptureKey(capture_batch_size=2),
        ),
        rows=(
            StageBatchRow(0, "request-1", 0, 0, compatibility),
            StageBatchRow(1, None, None, 0, compatibility),
        ),
    )
    return ScheduleDecision(decision_id, (batch,))


def _predictor_inputs(*, batch_id: int = 1):
    from nari_qwen3_tts.executor.rows import CodePredictorExecutionRow
    from nari_qwen3_tts.executor.submission import StageExecutionInputs

    row = CodePredictorExecutionRow(
        layer0_token=torch.tensor(3, dtype=torch.long),
        past_hidden=torch.zeros(4),
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=7,
        offsets=tuple(range(15)),
    )
    return StageExecutionInputs(batch_id=batch_id, rows=(row,))


def _execution(runtime):
    from nari_qwen3_tts.executor.executor import Executor

    return Executor(
        None,
        None,
        SimpleNamespace(replay=runtime.replay_code_predictor),
        SimpleNamespace(replay=runtime.replay_code_predictor),
        SimpleNamespace(replay=runtime.replay_code_predictor),
        None,
    )


def test_health_caches_the_immutable_capture_topology() -> None:
    from nari_qwen3_tts.contract import CodePredictorCaptureKey
    from nari_qwen3_tts.executor.executor import Executor

    class _CountingKeys(frozenset):
        iterations = 0

        def __iter__(self):
            type(self).iterations += 1
            return super().__iter__()

    keys = _CountingKeys((CodePredictorCaptureKey(1), CodePredictorCaptureKey(2)))
    stage = SimpleNamespace(
        capture_instances_per_key=1,
        captured_cuda_graph_instances=2,
    )
    execution = Executor(None, keys, stage, stage, stage, None)
    initialization_iterations = keys.iterations

    first = execution.health()
    second = execution.health()

    assert initialization_iterations == 1
    assert keys.iterations == initialization_iterations
    assert first.required_keys == second.required_keys == 2
    assert first.required_cuda_graph_instances == second.required_cuda_graph_instances == 2


def test_preflight_validates_the_whole_decision_without_replay_or_event_work() -> None:
    from nari_qwen3_tts.executor.types import CodePredictorResult

    runtime = _Runtime(
        CodePredictorResult(torch.zeros((1, 16), dtype=torch.long), torch.zeros((1, 4))),
        [],
    )
    execution = _execution(runtime)
    decision = _predictor_decision()
    inputs = _predictor_inputs()

    execution.preflight(decision, (inputs,))

    assert runtime.calls == []


def test_preflight_rejects_batch_identity_and_logical_row_mismatch() -> None:
    runtime = _Runtime(None, [])
    execution = _execution(runtime)
    decision = _predictor_decision()

    with pytest.raises(ValueError, match="batch ID"):
        execution.preflight(decision, (_predictor_inputs(batch_id=2),))
    with pytest.raises(ValueError, match="logical rows"):
        from nari_qwen3_tts.executor.submission import StageExecutionInputs

        source = _predictor_inputs().rows[0]
        execution.preflight(
            decision,
            (StageExecutionInputs(batch_id=1, rows=(source, source)),),
        )
    assert runtime.calls == []


def test_submit_preflights_before_replay_and_returns_typed_completion_fence() -> None:
    from nari_qwen3_tts.executor.cuda_graph import CudaSubmissionFence
    from nari_qwen3_tts.executor.submission import StageExecutionSubmission
    from nari_qwen3_tts.executor.types import CodePredictorResult

    output = CodePredictorResult(
        torch.zeros((1, 16), dtype=torch.long),
        torch.zeros((1, 4)),
    )
    runtime = _Runtime(output, [])
    execution = _execution(runtime)
    decision = _predictor_decision()
    execution_inputs = _predictor_inputs()

    submissions = execution.submit(decision, (execution_inputs,))

    assert len(submissions) == 1
    submission = submissions[0]
    assert isinstance(submission, StageExecutionSubmission)
    assert submission.batch is decision.batches[0]
    assert submission.result is output
    assert isinstance(submission.completion_fence, CudaSubmissionFence)
    assert submission.completion_fence.ready() is True
    assert isinstance(submission.decision_fence, CudaSubmissionFence)
    assert submission.decision_fence.ready() is True
    assert submission.requires_host_finalize is False
    assert len(runtime.calls) == 1
    stage, key, values = runtime.calls[0]
    assert stage == "code_predictor"
    assert key == decision.batches[0].capture.key
    assert values.rows[0] is execution_inputs.rows[0]
    assert values.sampler_route is decision.batches[0].compatibility.sampler_route


def test_submit_preflighted_does_not_repeat_whole_decision_validation(monkeypatch) -> None:
    from nari_qwen3_tts.executor.types import CodePredictorResult

    output = CodePredictorResult(
        torch.zeros((1, 16), dtype=torch.long),
        torch.zeros((1, 4)),
    )
    runtime = _Runtime(output, [])
    execution = _execution(runtime)
    decision = _predictor_decision()
    inputs = (_predictor_inputs(),)
    execution.preflight(decision, inputs)

    def repeated_preflight(_self, _decision, _inputs) -> None:
        raise AssertionError("validated submission repeated preflight")

    monkeypatch.setattr(type(execution), "preflight", repeated_preflight)
    submissions = execution.submit_preflighted(decision, inputs)

    assert len(submissions) == 1
    assert len(runtime.calls) == 1


def test_submit_rejects_a_late_invalid_batch_before_any_replay() -> None:
    from nari_qwen3_tts.contract import ScheduleDecision
    from nari_qwen3_tts.executor.types import CodePredictorResult

    output = CodePredictorResult(
        torch.zeros((1, 16), dtype=torch.long),
        torch.zeros((1, 4)),
    )
    runtime = _Runtime(output, [])
    execution = _execution(runtime)
    first = _predictor_decision(batch_id=1, decision_id=3).batches[0]
    second = _predictor_decision(batch_id=2, decision_id=3).batches[0]
    decision = ScheduleDecision(3, (first, second))

    with pytest.raises(ValueError, match="batch ID"):
        execution.submit(
            decision,
            (_predictor_inputs(batch_id=1), _predictor_inputs(batch_id=999)),
        )
    assert runtime.calls == []


def test_empty_codec_terminal_is_an_explicit_metadata_submission() -> None:
    from nari_qwen3_tts.contract import (
        CodecBatchCompatibility,
        CodecExecutionMode,
        ScheduleDecision,
        StageBatchRow,
        StageExecutionBatch,
        SynthesisStage,
    )
    from nari_qwen3_tts.executor.rows import CodecMetadataExecutionRow
    from nari_qwen3_tts.executor.submission import StageExecutionInputs

    class Runtime:
        @staticmethod
        def replay_talker_prefill(key, values):
            raise AssertionError((key, values))

        @staticmethod
        def replay_code_predictor(key, values):
            raise AssertionError((key, values))

        @staticmethod
        def replay_codec(key, values):
            raise AssertionError((key, values))

    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.EMPTY,
        model_frames=0,
        input_frames=0,
        visible_frames=0,
        pcm_start_frame=0,
        producer_frames=0,
        terminal=True,
    )
    batch = StageExecutionBatch(
        batch_id=1,
        decision_id=1,
        stage=SynthesisStage.CODEC,
        compatibility=compatibility,
        capture=None,
        rows=(StageBatchRow(0, "request-1", 2, 7, compatibility),),
    )
    runtime = Runtime()
    execution = _execution(runtime)

    submissions = execution.submit(
        ScheduleDecision(1, (batch,)),
        (StageExecutionInputs(1, (CodecMetadataExecutionRow(),)),),
    )

    assert execution._metadata_actions == 1
    assert submissions[0].result.pcm.shape == (1, 0)
    assert submissions[0].result.terminal is True

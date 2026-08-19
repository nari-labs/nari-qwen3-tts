from __future__ import annotations

import torch

from nari_qwen3_tts.contract import SynthesisStage
from nari_qwen3_tts.engine.state import GenerationPhase

from .test_pipeline_loop import _request, _runtime


def _run_three_steps(admission_order: tuple[int, ...]):
    runtime, _execution = _runtime()
    for index in admission_order:
        runtime.admit(_request(index))
    steps = tuple(runtime.step(now_s=position / 10) for position in range(3))
    states = {
        f"request-{index}": runtime.request(f"request-{index}").committed_view()
        for index in admission_order
    }
    return steps, states


def test_singleton_batch_and_admission_permutation_are_request_equivalent() -> None:
    _batch_steps, batch = _run_three_steps((0, 1, 2))
    _permuted_steps, permuted = _run_three_steps((2, 0, 1))
    singleton = {
        request_id: _run_three_steps((index,))[1][request_id]
        for index, request_id in enumerate(("request-0", "request-1", "request-2"))
    }

    assert batch == permuted
    assert batch == singleton


def test_prefill_padding_manifest_never_enters_executor_request_rows() -> None:
    runtime, execution = _runtime()
    for index in range(3):
        runtime.admit(_request(index))
    step = runtime.step(now_s=0.0)
    plan = step.decision.batches[0]

    assert plan.stage is SynthesisStage.TALKER_PREFILL
    assert plan.logical_rows == 3
    assert len(plan.rows) == 4
    assert plan.padding_rows == 1
    assert all(row.padding and row.request_id is None for row in plan.rows[3:])
    assert execution.calls[-1][1] == ("request-0", "request-1", "request-2")


def test_capture_capacity_split_retains_each_cp_request_and_rng_address() -> None:
    runtime, execution = _runtime()
    for index in range(65):
        runtime_request = _request(index)
        runtime.admit(runtime_request)
        state = runtime.request(runtime_request.request_id)
        state.generation.phase = GenerationPhase.CODE_PREDICTOR
        state.generation.token = torch.tensor(index % 31)
        state.generation.hidden = torch.full((4,), float(index))

    step = runtime.step(now_s=0.0)

    assert step.decision.selected_stage is SynthesisStage.CODE_PREDICTOR
    assert [plan.logical_rows for plan in step.decision.batches] == [64, 1]
    assert [call[0] for call in execution.calls] == ["code_predictor_rows", "code_predictor_rows"]
    assert [len(call[1]) for call in execution.calls] == [64, 1]
    assert step.committed_request_ids == tuple(f"request-{index}" for index in range(65))
    for index in range(65):
        state = runtime.request(f"request-{index}")
        assert state.codec.buffered_frames[0][0].item() == index % 31
        assert state.generation.phase is GenerationPhase.TALKER_DECODE

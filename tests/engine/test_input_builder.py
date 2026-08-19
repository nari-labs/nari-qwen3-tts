from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch
from scheduler.test_pipeline_loop import _request, _runtime

from nari_qwen3_tts.contract import SynthesisStage
from nari_qwen3_tts.engine.input_builder import InputBuilder


def _assert_same(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
        return
    if is_dataclass(left):
        for field in fields(left):
            _assert_same(getattr(left, field.name), getattr(right, field.name))
        return
    if isinstance(left, tuple):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_same(left_item, right_item)
        return
    assert left == right


def test_input_builder_is_the_unique_prefill_input_authority() -> None:
    runtime, _execution = _runtime()
    request = _request(1)
    runtime.admit(request)
    candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    prepared = runtime._prepare_execution(decision)
    runtime._claim_decision(decision)
    submitted = prepared.inputs

    builder = InputBuilder(model_spec=runtime.model_config)
    canonical = builder.build(decision, runtime.state_store)

    _assert_same(canonical, submitted)


def test_input_builder_covers_all_four_stage_input_surfaces() -> None:
    runtime, _execution = _runtime()
    runtime.admit(_request(1))
    observed: set[SynthesisStage] = set()

    for turn in range(32):
        candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=float(turn))
        if candidates:
            decision = runtime.planner.plan(
                candidates,
                now_s=float(turn),
                decision_id=runtime._next_decision_id(),
                available_rows=runtime.state_store.available_in_flight_rows,
            )
            prepared = runtime._prepare_execution(decision)
            claim = runtime._claim_decision(decision)
            submitted = prepared.inputs
            canonical = runtime.input_builder.build(decision, runtime.state_store)
            _assert_same(canonical, submitted)
            observed.update(batch.stage for batch in decision.batches)
            runtime.execute_decision(prepared, claim)
        if observed == set(SynthesisStage):
            break

    assert observed == set(SynthesisStage)


def test_prepare_execution_materializes_each_request_state_once(monkeypatch) -> None:
    runtime, _execution = _runtime()
    runtime.admit(_request(0))
    runtime.admit(_request(2))
    candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    calls: dict[str, int] = {}
    original_request = runtime.state_store.request

    def request_once(request_id: str):
        calls[request_id] = calls.get(request_id, 0) + 1
        return original_request(request_id)

    monkeypatch.setattr(runtime.state_store, "request", request_once)

    prepared = runtime._prepare_execution(decision)

    assert tuple(values.batch_id for values in prepared.inputs) == tuple(
        batch.batch_id for batch in decision.batches
    )
    expected_request_ids = {
        request_id
        for batch in decision.batches
        for request_id in batch.request_ids
    }
    assert calls == {request_id: 1 for request_id in expected_request_ids}

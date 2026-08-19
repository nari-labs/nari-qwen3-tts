from __future__ import annotations

from dataclasses import replace

from scheduler.test_pipeline_loop import _request, _runtime


def test_engine_submits_one_token_free_canonical_decision() -> None:
    from nari_qwen3_tts.contract import ScheduleDecision, StageExecutionBatch

    runtime, execution = _runtime()
    runtime.admit(_request(0))
    candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )
    prepared = runtime._prepare_execution(decision)
    claim = runtime._claim_decision(decision)
    inputs = prepared.inputs
    contexts = tuple((batch, states, claim) for batch, states in prepared.batch_states)

    assert isinstance(decision, ScheduleDecision)
    assert all(isinstance(batch, StageExecutionBatch) for batch in decision.batches)
    assert runtime.committer.batches(claim) == decision.batches
    assert tuple(values.batch_id for values in inputs) == tuple(
        batch.batch_id for batch in decision.batches
    )
    assert tuple(context[0] for context in contexts) == decision.batches
    assert all(context[2] is claim for context in contexts)
    assert execution.calls == []


def test_runtime_submission_input_preserves_streaming_plan_reuse_and_eos_visibility() -> None:
    runtime, _execution = _runtime()
    request = _request(1)
    request = replace(
        request,
        request=replace(request.request, max_new_tokens=4),
    )
    runtime.admit(request)

    for index in range(16):
        candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=float(index))
        decision = runtime.planner.plan(
            candidates,
            now_s=float(index),
            decision_id=runtime._next_decision_id(),
            available_rows=runtime.state_store.available_in_flight_rows,
        )
        prepared = runtime._prepare_execution(decision)
        claim = runtime._claim_decision(decision)
        inputs = prepared.inputs
        if all(values.requires_host_finalize for values in inputs):
            assert all(values.reuse_attention_plan for values in inputs)
            break
        runtime.execute_decision(prepared, claim)
    else:
        raise AssertionError("streaming decode never reached natural-EOS host visibility")


def test_submission_window_allows_later_ready_work_to_bypass() -> None:
    from nari_qwen3_tts.executor.submission import SubmissionWindow

    class Fence:
        def __init__(self, ready: bool) -> None:
            self.done = ready
            self.waits = 0

        def ready(self) -> bool:
            return self.done

        def wait(self) -> None:
            self.waits += 1
            self.done = True

    class Submission:
        def __init__(self, completion, decision, plan_id: int) -> None:
            self.completion_fence = completion
            self.decision_fence = decision
            self.requires_host_finalize = True
            self.plan_id = plan_id

    first = Fence(False)
    second = Fence(True)
    window = SubmissionWindow(max_decisions=2)
    window.record(
        decision_id=1,
        submissions=(Submission(first, Fence(True), 11),),
    )
    window.record(
        decision_id=2,
        submissions=(Submission(second, Fence(True), 12),),
    )

    assert tuple(item.plan_id for item in window.poll_host_ready(block_oldest=False)) == (12,)
    assert tuple(item.plan_id for item in window.poll_host_ready(block_oldest=True)) == (11,)
    assert first.waits == 1

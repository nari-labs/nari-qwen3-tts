from __future__ import annotations

from types import SimpleNamespace

import torch
from scheduler.test_pipeline_loop import _direct_runtime, _request, _runtime

import nari_qwen3_tts.executor.cuda_graph as cuda_graph_module
from nari_qwen3_tts.contract import SynthesisStage
from nari_qwen3_tts.executor import CodePredictorExecutor as CodePredictorCudaExecutor
from nari_qwen3_tts.executor import CudaGraphPoolFence


def test_runtime_keeps_at_most_two_decision_fences_in_flight(monkeypatch) -> None:
    del monkeypatch
    runtime, _execution = _direct_runtime()

    assert runtime._submissions.max_decisions == 2
    assert runtime._submissions.can_submit is True


def test_immediate_stage_completions_do_not_create_host_events() -> None:
    runtime, _execution = _runtime()
    created: list[object] = []
    runtime.completion_event_factory = lambda: created.append(object()) or created[-1]
    runtime.admit(_request(0))

    stages = [runtime.step(now_s=float(index)).decision.selected_stage for index in range(3)]

    assert stages == [
        SynthesisStage.TALKER_PREFILL,
        SynthesisStage.CODE_PREDICTOR,
        SynthesisStage.CODEC,
    ]
    assert created == []


def test_decision_partial_order_is_claim_replay_fence_then_commit(monkeypatch) -> None:
    runtime, execution = _direct_runtime()
    runtime.admit(_request(0))
    events: list[str] = []

    original_claim = runtime.committer.claim
    original_preflight = execution.preflight
    original_replay = execution.talker_prefill_rows
    original_commit = runtime.committer.apply
    original_submit = execution.submit_preflighted

    def preflight(decision, inputs):
        events.append("preflight")
        return original_preflight(decision, inputs)

    def claim(decision):
        events.append("claim")
        return original_claim(decision)

    def replay(key, values):
        events.append("replay")
        return original_replay(key, values)

    def commit(*args, **kwargs):
        events.append("commit")
        return original_commit(*args, **kwargs)

    def submit(decision, inputs):
        result = original_submit(decision, inputs)
        events.append("decision_fence")
        return result

    monkeypatch.setattr(execution, "preflight", preflight)
    monkeypatch.setattr(runtime.committer, "claim", claim)
    monkeypatch.setattr(execution, "talker_prefill_rows", replay)
    monkeypatch.setattr(runtime.committer, "apply", commit)
    monkeypatch.setattr(execution, "submit_preflighted", submit)

    runtime.step(now_s=0.0)

    assert events == ["preflight", "claim", "replay", "decision_fence", "commit"]


def test_planning_does_not_claim_rows_or_call_the_executor() -> None:
    runtime, execution = _direct_runtime()
    runtime.admit(_request(0))
    before = runtime.request("request-0").committed_view()

    candidates = runtime.planner.candidates(runtime.state_store.requests, now_s=0.0)
    decision = runtime.planner.plan(
        candidates,
        now_s=0.0,
        decision_id=runtime._next_decision_id(),
        available_rows=runtime.state_store.available_in_flight_rows,
    )

    assert decision.selected_stage is SynthesisStage.TALKER_PREFILL
    assert execution.calls == []
    assert runtime.request("request-0").committed_view() == before
    assert runtime.request("request-0").generation.claim_token is None
    runtime.planner.discarded(decision)


def test_shared_cuda_pool_records_completion_before_next_submission_waits(monkeypatch) -> None:
    events: list[str] = []

    class _Event:
        def record(self, stream) -> None:
            del stream
            events.append("record")

    class _Stream:
        def wait_event(self, event) -> None:
            assert isinstance(event, _Event)
            events.append("wait")

    def event_factory(*, blocking):
        assert blocking is False
        return _Event()

    stream = _Stream()
    monkeypatch.setattr(cuda_graph_module.torch.cuda, "Event", event_factory)
    monkeypatch.setattr(cuda_graph_module.torch.cuda, "current_stream", lambda _device: stream)
    fence = CudaGraphPoolFence(device=torch.device("cuda"))

    first = fence.reserve()
    fence.release(first)
    second = fence.reserve()
    fence.release(second)

    assert events == ["record", "wait", "record"]


def test_code_predictor_reuses_written_before_read_kv_without_clearing(monkeypatch) -> None:
    config = SimpleNamespace(
        num_code_groups=16,
        talker=SimpleNamespace(hidden_size=4),
        code_predictor=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=4,
        ),
    )
    job = CodePredictorCudaExecutor(
        model=SimpleNamespace(small_to_mtp_projection=torch.nn.Linear(4, 4)),
        layer0_embedding=torch.nn.Embedding(4, 4),
        config=config,
        max_batch_size=4,
        driver=object(),
    )
    reference = torch.ones((), dtype=torch.float32)
    cache = job._cache(4, reference)
    cache.fill_(7)
    zeroed: list[int] = []
    original_zero = torch.Tensor.zero_

    def zero_spy(value, *args, **kwargs):
        if value.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr():
            zeroed.append(value.data_ptr())
        return original_zero(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "zero_", zero_spy)

    reused = job._cache(2, reference)

    assert reused.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.all(reused == 7)
    assert zeroed == []

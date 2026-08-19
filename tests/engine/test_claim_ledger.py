from __future__ import annotations

import pytest

from nari_qwen3_tts.contract import (
    TALKER_DECODE_COMPATIBILITY,
    CodecBatchCompatibility,
    CodecCaptureKey,
    CodecExecutionMode,
    CodecStateDelta,
    CudaGraphRef,
    IncrementalCodecState,
    ScheduleDecision,
    StageBatchRow,
    StageBatchRowResult,
    StageExecutionBatch,
    StageExecutionCompletion,
    SynthesisStage,
    TalkerDecodeCaptureKey,
)
from nari_qwen3_tts.engine.state import (
    CodecPhase,
    DuplicateCompletionError,
    EngineStateError,
    GenerationPhase,
    RequestState,
    RequestStateStore,
)


def _in_flight_codec_completion(*, buffered_frames: int):
    from nari_qwen3_tts.engine.commit import Committer

    compatibility = CodecBatchCompatibility(
        mode=CodecExecutionMode.WARM,
        model_frames=12,
        input_frames=12,
        visible_frames=12,
        pcm_start_frame=0,
        producer_frames=12,
        terminal=False,
    )
    row = StageBatchRow(
        physical_row=0,
        request_id="codec",
        version=0,
        logical_step=4,
        compatibility=compatibility,
    )
    batch = StageExecutionBatch(
        batch_id=1,
        decision_id=1,
        stage=SynthesisStage.CODEC,
        compatibility=compatibility,
        capture=CudaGraphRef(
            SynthesisStage.CODEC,
            CodecCaptureKey(CodecExecutionMode.WARM, 12, 1),
        ),
        rows=(row,),
    )
    store = RequestStateStore(max_in_flight_rows=1)
    state = RequestState.for_testing("codec")
    state.codec.phase = CodecPhase.READY
    state.codec.chunk_index = 4
    state.codec.buffered_frames = tuple(object() for _ in range(buffered_frames))
    state.codec.decoder_state = IncrementalCodecState()
    state.codec.ready_compatibility = compatibility
    store.admit(state)
    committer = Committer(store)
    handle = committer.claim(ScheduleDecision(1, (batch,)))

    # The generation lane may finish while this non-terminal Codec batch is
    # already in flight.  Readiness cannot be refreshed until its claim is
    # released, because the claimed static output is not committed yet.
    state.codec.producer_done = True
    completion = StageExecutionCompletion(
        batch_id=1,
        stage=SynthesisStage.CODEC,
        rows=(
            StageBatchRowResult(
                row,
                CodecStateDelta(
                    state=IncrementalCodecState(),
                    consumed_frames=12,
                    visible_frames=12,
                    terminal=False,
                ),
            ),
        ),
    )
    committer.apply(handle, completion)
    return state


def _decision(*, second_version: int = 0) -> ScheduleDecision:
    capture = CudaGraphRef(
        SynthesisStage.TALKER_DECODE,
        TalkerDecodeCaptureKey(1),
    )
    batches = tuple(
        StageExecutionBatch(
            batch_id=index + 1,
            decision_id=1,
            stage=SynthesisStage.TALKER_DECODE,
            compatibility=TALKER_DECODE_COMPATIBILITY,
            capture=capture,
            rows=(
                StageBatchRow(
                    physical_row=0,
                    request_id=request_id,
                    version=version,
                    logical_step=0,
                    compatibility=TALKER_DECODE_COMPATIBILITY,
                ),
            ),
        )
        for index, (request_id, version) in enumerate(
            (("a", 0), ("b", second_version))
        )
    )
    return ScheduleDecision(1, batches)


def _store() -> RequestStateStore:
    store = RequestStateStore(max_in_flight_rows=2)
    for request_id in ("a", "b"):
        state = RequestState.for_testing(request_id)
        state.generation.phase = GenerationPhase.TALKER_DECODE
        store.admit(state)
    return store


def test_engine_private_claim_assigns_tokens_only_after_all_rows_validate() -> None:
    from nari_qwen3_tts.engine.commit import Committer

    store = _store()
    committer = Committer(store)
    decision = _decision()

    handle = committer.claim(decision)
    batches = committer.batches(handle)

    assert tuple(batch.batch_id for batch in batches) == (1, 2)
    assert tuple(batch.request_ids for batch in batches) == (("a",), ("b",))
    assert all("token" not in batch.rows[0].__dataclass_fields__ for batch in batches)
    assert store.in_flight_rows == 2
    assert store.request("a").generation.claim_batch_id == 1
    assert store.request("b").generation.claim_batch_id == 2


def test_claim_failure_is_atomic_and_does_not_publish_a_partial_handle() -> None:
    from nari_qwen3_tts.engine.commit import Committer

    store = _store()
    committer = Committer(store)

    with pytest.raises(EngineStateError, match="version"):
        committer.claim(_decision(second_version=1))

    assert store.in_flight_rows == 0
    assert store.request("a").generation.claim_token is None
    assert store.request("b").generation.claim_token is None
    assert committer.active_handles == ()


def test_claim_handle_rejects_duplicate_claim_and_double_release() -> None:
    from nari_qwen3_tts.engine.commit import Committer

    store = _store()
    committer = Committer(store)
    decision = _decision()
    handle = committer.claim(decision)

    with pytest.raises(EngineStateError, match="already claimed"):
        committer.claim(decision)

    committer.reject(handle, batch_id=1, error=RuntimeError("rejected"))
    with pytest.raises(DuplicateCompletionError, match="already consumed"):
        committer.reject(handle, batch_id=1, error=RuntimeError("duplicate"))
    committer.reject(handle, batch_id=2, error=RuntimeError("rejected"))
    assert committer.active_handles == ()


def test_claim_validation_rejects_claimed_lane_without_changing_other_rows() -> None:
    from nari_qwen3_tts.engine.commit import Committer

    store = _store()
    store.request("b").generation.claim_token = 999
    store.request("b").generation.claim_batch_id = 999
    committer = Committer(store)

    with pytest.raises(EngineStateError, match="in-flight claim"):
        committer.claim(_decision())

    assert store.request("a").generation.claim_token is None
    assert store.request("b").generation.claim_token == 999
    assert committer.active_handles == ()


def test_engine_step_uses_planner_and_releases_private_claims() -> None:
    from scheduler.test_pipeline_loop import _request, _runtime

    runtime, _execution = _runtime()
    runtime.admit(_request(0))

    result = runtime.step(now_s=0.0)

    assert result is not None
    assert result.decision.selected_stage is SynthesisStage.TALKER_PREFILL
    assert runtime.committer.active_handles == ()


def test_runtime_preflight_failure_does_not_create_a_private_claim(monkeypatch) -> None:
    from scheduler.test_pipeline_loop import _request, _runtime

    runtime, execution = _runtime()
    runtime.admit(_request(0))
    claim_calls = 0
    original_claim = runtime.committer.claim

    def count_claim(decision):
        nonlocal claim_calls
        claim_calls += 1
        return original_claim(decision)

    def fail_preflight(_decision, _inputs) -> None:
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(execution, "preflight", fail_preflight)
    monkeypatch.setattr(runtime.committer, "claim", count_claim)

    with pytest.raises(RuntimeError, match="preflight failed"):
        runtime.step(now_s=0.0)

    state = runtime.request("request-0")
    assert state.generation.claim_token is None
    assert state.generation.claim_batch_id is None
    assert runtime.committer.active_handles == ()
    assert claim_calls == 0


def test_successful_codec_release_refreshes_terminal_partial_after_generation_finishes() -> None:
    state = _in_flight_codec_completion(buffered_frames=13)

    assert state.codec.claim_token is None
    assert state.codec.claim_batch_id is None
    assert len(state.codec.buffered_frames) == 1
    assert state.codec.phase is CodecPhase.READY
    assert state.codec.ready_compatibility == CodecBatchCompatibility(
        mode=CodecExecutionMode.WARM,
        model_frames=12,
        input_frames=1,
        visible_frames=1,
        pcm_start_frame=0,
        producer_frames=1,
        terminal=True,
    )


def test_successful_codec_release_refreshes_empty_terminal_after_generation_finishes() -> None:
    state = _in_flight_codec_completion(buffered_frames=12)

    assert state.codec.claim_token is None
    assert state.codec.claim_batch_id is None
    assert state.codec.buffered_frames == ()
    assert state.codec.phase is CodecPhase.READY
    assert state.codec.ready_compatibility == CodecBatchCompatibility(
        mode=CodecExecutionMode.EMPTY,
        model_frames=0,
        input_frames=0,
        visible_frames=0,
        pcm_start_frame=0,
        producer_frames=0,
        terminal=True,
    )

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nari_qwen3_tts.contract import (
    TALKER_DECODE_COMPATIBILITY,
    CodecBatchCompatibility,
    CodecCaptureKey,
    CodecExecutionMode,
    CodecStateDelta,
    CodePredictorBatchCompatibility,
    CodePredictorCaptureKey,
    CodePredictorStateDelta,
    CudaGraphRef,
    IncrementalCodecState,
    KVPublication,
    ScheduleDecision,
    StageBatchRow,
    StageBatchRowResult,
    StageExecutionBatch,
    StageExecutionCompletion,
    SynthesisStage,
    TalkerDecodeCaptureKey,
    TalkerStateDelta,
)
from nari_qwen3_tts.engine.commit import Committer
from nari_qwen3_tts.engine.state import (
    CodecPhase,
    DuplicateCompletionError,
    EngineStateError,
    GenerationPhase,
    RequestState,
    RequestStateStore,
    StaleCompletionError,
)


class _Publication:
    def __init__(self, request_id: str, *, valid: bool = True, fail_publish: bool = False) -> None:
        self.request_id = request_id
        self.valid = valid
        self.fail_publish = fail_publish
        self.published = 0
        self.discarded = 0

    def validate(self) -> None:
        if not self.valid:
            raise RuntimeError("invalid publication")

    def publish(self) -> None:
        self.validate()
        if self.fail_publish:
            raise RuntimeError("publication failed")
        self.published += 1

    def discard(self) -> None:
        self.discarded += 1


def _ready(request_id: str) -> RequestState:
    state = RequestState.for_testing(request_id)
    state.generation.phase = GenerationPhase.TALKER_DECODE
    state.generation.step_input = torch.ones(4)
    return state


def _decision(store: RequestStateStore, request_ids: tuple[str, ...]) -> ScheduleDecision:
    size = len(request_ids)
    batch = StageExecutionBatch(
        batch_id=1,
        decision_id=1,
        stage=SynthesisStage.TALKER_DECODE,
        compatibility=TALKER_DECODE_COMPATIBILITY,
        capture=CudaGraphRef(
            SynthesisStage.TALKER_DECODE,
            TalkerDecodeCaptureKey(size),
        ),
        rows=tuple(
            StageBatchRow(
                physical_row=index,
                request_id=request_id,
                version=store.request(request_id).version,
                logical_step=store.request(request_id).generation.generation_step,
                compatibility=TALKER_DECODE_COMPATIBILITY,
            )
            for index, request_id in enumerate(request_ids)
        ),
    )
    return ScheduleDecision(1, (batch,))


def _delta(request_id: str, token: int = 3, *, offset: int = 512) -> TalkerStateDelta:
    return TalkerStateDelta(
        token=torch.tensor(token, dtype=torch.long),
        hidden=torch.tensor([float(token)]),
        logits=torch.tensor([float(token), 0.0]),
        next_seen_token_mask=torch.tensor([False, True]),
        next_sampling_offset=offset,
        kv=KVPublication(request_id, (token,), 1),
    )


def _completion(
    decision: ScheduleDecision,
    *,
    rows: tuple[StageBatchRow, ...] | None = None,
) -> StageExecutionCompletion:
    batch = decision.batches[0]
    selected = batch.rows if rows is None else rows
    return StageExecutionCompletion(
        batch_id=batch.batch_id,
        stage=batch.stage,
        rows=tuple(
            StageBatchRowResult(row, _delta(row.request_id or "padding"))
            for row in selected
        ),
    )


def _store(*request_ids: str, capacity: int = 8) -> RequestStateStore:
    store = RequestStateStore(max_in_flight_rows=capacity)
    for request_id in request_ids:
        store.admit(_ready(request_id))
    return store


def _assert_released(store: RequestStateStore, *request_ids: str) -> None:
    assert store.in_flight_rows == 0
    for request_id in request_ids:
        lane = store.request(request_id).generation
        assert lane.claim_token is None
        assert lane.claim_batch_id is None


def test_claim_is_atomic_and_does_not_publish_request_progress() -> None:
    store = _store("a", "b")
    decision = _decision(store, ("a", "b"))
    before = tuple(store.request(request_id).committed_view() for request_id in ("a", "b"))

    handle = Committer(store).claim(decision)

    assert tuple(store.request(request_id).committed_view() for request_id in ("a", "b")) == before
    assert handle.batch_ids == (1,)
    assert store.in_flight_rows == 2


def test_stale_claim_rejects_all_rows_before_assigning_any_owner() -> None:
    store = _store("a", "b")
    decision = _decision(store, ("a", "b"))
    batch = decision.batches[0]
    stale = replace(batch.rows[1], version=99)
    decision = replace(decision, batches=(replace(batch, rows=(batch.rows[0], stale)),))

    with pytest.raises(StaleCompletionError, match="version"):
        Committer(store).claim(decision)

    _assert_released(store, "a", "b")


def test_completion_preflights_every_row_and_publication_before_state_publish() -> None:
    store = _store("a", "b")
    decision = _decision(store, ("a", "b"))
    committer = Committer(store)
    handle = committer.claim(decision)
    publications = (_Publication("a"), _Publication("b", valid=False))

    with pytest.raises(RuntimeError, match="invalid publication"):
        committer.apply(
            handle,
            _completion(decision),
            kv_publications=publications,
            talker_terminal=(False, False),
        )

    _assert_released(store, "a", "b")
    assert store.request("a").version == store.request("b").version == 0
    assert all(publication.published == 0 for publication in publications)
    assert all(publication.discarded == 1 for publication in publications)


def test_stale_completion_row_releases_claim_and_discards_every_publication() -> None:
    store = _store("a", "b")
    decision = _decision(store, ("a", "b"))
    committer = Committer(store)
    handle = committer.claim(decision)
    rows = decision.batches[0].rows
    stale_rows = (rows[0], replace(rows[1], version=99))
    publications = (_Publication("a"), _Publication("b"))

    with pytest.raises(StaleCompletionError, match="physical rows"):
        committer.apply(
            handle,
            _completion(decision, rows=stale_rows),
            kv_publications=publications,
            talker_terminal=(False, False),
        )

    _assert_released(store, "a", "b")
    assert all(publication.discarded == 1 for publication in publications)


def test_cancelled_peer_rejects_a_whole_batch_without_partial_publish() -> None:
    store = _store("a", "b")
    decision = _decision(store, ("a", "b"))
    committer = Committer(store)
    handle = committer.claim(decision)
    store.cancel("b")
    publications = (_Publication("a"), _Publication("b"))

    assert committer.apply(
        handle,
        _completion(decision),
        kv_publications=publications,
        talker_terminal=(False, False),
    ) == ()

    _assert_released(store, "a", "b")
    assert store.request("a").version == store.request("b").version == 0
    assert all(publication.published == 0 for publication in publications)


def test_request_local_kv_and_rng_are_validated_before_publication() -> None:
    store = _store("a")
    decision = _decision(store, ("a",))
    committer = Committer(store)
    handle = committer.claim(decision)
    publication = _Publication("other")

    with pytest.raises(StaleCompletionError, match="publication belongs"):
        committer.apply(
            handle,
            _completion(decision),
            kv_publications=(publication,),
            talker_terminal=(False,),
        )

    _assert_released(store, "a")
    assert publication.published == 0

    original = _decision(store, ("a",))
    decision = ScheduleDecision(
        2,
        (replace(original.batches[0], decision_id=2, batch_id=2),),
    )
    handle = committer.claim(decision)
    row = decision.batches[0].rows[0]
    completion = StageExecutionCompletion(
        batch_id=2,
        stage=SynthesisStage.TALKER_DECODE,
        rows=(StageBatchRowResult(row, _delta("a", offset=1024)),),
    )
    with pytest.raises(StaleCompletionError, match="sampling offset"):
        committer.apply(
            handle,
            completion,
            kv_publications=(_Publication("a"),),
            talker_terminal=(False,),
        )


def test_valid_completion_publishes_once_and_duplicate_is_rejected() -> None:
    store = _store("a")
    decision = _decision(store, ("a",))
    committer = Committer(store)
    handle = committer.claim(decision)
    publication = _Publication("a")
    completion = _completion(decision)

    assert committer.apply(
        handle,
        completion,
        kv_publications=(publication,),
        talker_terminal=(False,),
    ) == ("a",)

    state = store.request("a")
    assert state.version == 1
    assert state.generation.phase is GenerationPhase.CODE_PREDICTOR
    assert state.generation.next_sampling_offset == 512
    assert publication.published == 1
    with pytest.raises((DuplicateCompletionError, EngineStateError)):
        committer.apply(
            handle,
            completion,
            kv_publications=(publication,),
            talker_terminal=(False,),
        )
    assert publication.published == 1


def test_claim_capacity_is_reclaimed_across_more_requests_than_capacity() -> None:
    store = RequestStateStore(max_in_flight_rows=2)
    committer = Committer(store)
    for index in range(8):
        request_id = f"request-{index}"
        store.admit(_ready(request_id))
        decision = _decision(store, (request_id,))
        batch = replace(decision.batches[0], batch_id=index + 1, decision_id=index + 1)
        decision = ScheduleDecision(index + 1, (batch,))
        handle = committer.claim(decision)
        store.cancel(request_id)
        committer.apply(
            handle,
            StageExecutionCompletion(
                batch_id=batch.batch_id,
                stage=batch.stage,
                rows=(StageBatchRowResult(batch.rows[0], _delta(request_id)),),
            ),
            kv_publications=(_Publication(request_id),),
            talker_terminal=(False,),
        )
        store.remove(request_id)
    assert store.in_flight_rows == 0
    assert committer.active_handles == ()
    assert committer._completed_batches == set()


def _stage_batch(
    state: RequestState,
    *,
    batch_id: int,
    decision_id: int,
    stage: SynthesisStage,
    compatibility,
    capture,
) -> StageExecutionBatch:
    logical_step = (
        state.codec.chunk_index
        if stage is SynthesisStage.CODEC
        else state.generation.generation_step
    )
    return StageExecutionBatch(
        batch_id=batch_id,
        decision_id=decision_id,
        stage=stage,
        compatibility=compatibility,
        capture=capture,
        rows=(
            StageBatchRow(
                physical_row=0,
                request_id=state.request_id,
                version=state.version_for(stage),
                logical_step=logical_step,
                compatibility=compatibility,
            ),
        ),
    )


def test_cold_codec_commit_releases_bootstrap_history() -> None:
    store = RequestStateStore()
    state = _ready("request")
    state.generation.phase = GenerationPhase.DONE
    state.codec.phase = CodecPhase.READY
    frames = tuple(torch.arange(16) + index for index in range(4))
    state.codec.buffered_frames = frames
    state.codec.history_frames = frames
    state.codec.producer_done = True
    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.COLD,
        model_frames=4,
        input_frames=4,
        visible_frames=4,
        pcm_start_frame=0,
        producer_frames=4,
        terminal=True,
    )
    state.codec.ready_compatibility = compatibility
    store.admit(state)
    batch = _stage_batch(
        state,
        batch_id=1,
        decision_id=1,
        stage=SynthesisStage.CODEC,
        compatibility=compatibility,
        capture=CudaGraphRef(
            SynthesisStage.CODEC,
            CodecCaptureKey(CodecExecutionMode.COLD, 4, 1),
        ),
    )
    committer = Committer(store)
    handle = committer.claim(ScheduleDecision(1, (batch,)))
    delta = CodecStateDelta(IncrementalCodecState(), 4, 4, True)

    committer.apply(
        handle,
        StageExecutionCompletion(
            batch.batch_id,
            batch.stage,
            (StageBatchRowResult(batch.rows[0], delta),),
        ),
    )

    assert state.codec.decoder_state is not None
    assert state.codec.history_frames == ()


def _run_concurrent_cp_and_codec(*, codec_first: bool) -> tuple[object, ...]:
    store = RequestStateStore()
    state = _ready("request")
    state.generation.phase = GenerationPhase.CODE_PREDICTOR
    state.generation.hidden = torch.ones(4)
    original = torch.arange(16)
    state.codec.buffered_frames = (original,)
    state.codec.history_frames = (original,)
    state.codec.phase = CodecPhase.READY
    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.WHOLE_SEQUENCE,
        model_frames=1,
        input_frames=1,
        visible_frames=1,
        pcm_start_frame=0,
        producer_frames=1,
        terminal=False,
    )
    state.codec.ready_compatibility = compatibility
    store.admit(state)
    cp_batch = _stage_batch(
        state,
        batch_id=1,
        decision_id=1,
        stage=SynthesisStage.CODE_PREDICTOR,
        compatibility=CodePredictorBatchCompatibility(),
        capture=CudaGraphRef(
            SynthesisStage.CODE_PREDICTOR,
            CodePredictorCaptureKey(1),
        ),
    )
    codec_batch = _stage_batch(
        state,
        batch_id=2,
        decision_id=1,
        stage=SynthesisStage.CODEC,
        compatibility=compatibility,
        capture=CudaGraphRef(
            SynthesisStage.CODEC,
            CodecCaptureKey(CodecExecutionMode.WHOLE_SEQUENCE, 1, 1),
        ),
    )
    committer = Committer(store)
    handle = committer.claim(ScheduleDecision(1, (cp_batch, codec_batch)))
    new_frame = torch.arange(16) + 100
    completions = {
        SynthesisStage.CODE_PREDICTOR: StageExecutionCompletion(
            cp_batch.batch_id,
            cp_batch.stage,
            (
                StageBatchRowResult(
                    cp_batch.rows[0],
                    CodePredictorStateDelta(new_frame, torch.full((4,), 4.0)),
                ),
            ),
        ),
        SynthesisStage.CODEC: StageExecutionCompletion(
            codec_batch.batch_id,
            codec_batch.stage,
            (
                StageBatchRowResult(
                    codec_batch.rows[0],
                    CodecStateDelta(None, 1, 1, False),
                ),
            ),
        ),
    }
    order = (
        (SynthesisStage.CODEC, SynthesisStage.CODE_PREDICTOR)
        if codec_first
        else (SynthesisStage.CODE_PREDICTOR, SynthesisStage.CODEC)
    )
    for stage in order:
        committer.apply(handle, completions[stage])
    assert len(state.codec.buffered_frames) == 1
    assert torch.equal(state.codec.buffered_frames[0], new_frame)
    return state.committed_view()


def test_generation_and_codec_lane_commits_are_order_independent() -> None:
    assert _run_concurrent_cp_and_codec(codec_first=True) == _run_concurrent_cp_and_codec(
        codec_first=False
    )


def test_empty_terminal_metadata_commits_exactly_once() -> None:
    store = RequestStateStore()
    state = _ready("request")
    state.generation.phase = GenerationPhase.DONE
    state.codec.phase = CodecPhase.READY
    state.codec.producer_done = True
    compatibility = CodecBatchCompatibility(
        CodecExecutionMode.EMPTY,
        model_frames=0,
        input_frames=0,
        visible_frames=0,
        pcm_start_frame=0,
        producer_frames=0,
        terminal=True,
    )
    state.codec.ready_compatibility = compatibility
    store.admit(state)
    batch = _stage_batch(
        state,
        batch_id=1,
        decision_id=1,
        stage=SynthesisStage.CODEC,
        compatibility=compatibility,
        capture=None,
    )
    committer = Committer(store)
    handle = committer.claim(ScheduleDecision(1, (batch,)))
    completion = StageExecutionCompletion(
        batch.batch_id,
        batch.stage,
        (StageBatchRowResult(batch.rows[0], CodecStateDelta(None, 0, 0, True)),),
    )

    assert committer.apply(handle, completion) == ("request",)
    assert state.codec.phase is CodecPhase.DONE
    assert state.codec.compute_terminal
    with pytest.raises(EngineStateError):
        committer.apply(handle, completion)

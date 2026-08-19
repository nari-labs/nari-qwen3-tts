"""Engine-private claim ledger and atomic canonical completion commit."""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.contract.result import (
    CodecStateDelta,
    CodePredictorStateDelta,
    StageBatchRow,
    StageExecutionCompletion,
    TalkerStateDelta,
)
from nari_qwen3_tts.contract.stage import CodecExecutionMode, SynthesisStage
from nari_qwen3_tts.contract.work import (
    CodecBatchCompatibility,
    ScheduleDecision,
    StageExecutionBatch,
)
from nari_qwen3_tts.engine.codec_readiness import next_codec_readiness
from nari_qwen3_tts.engine.state import (
    TALKER_RNG_FRAME_STRIDE,
    CodecPhase,
    DuplicateCompletionError,
    EngineStateError,
    GenerationPhase,
    RequestState,
    RequestStateStore,
    ResourceExhaustedError,
    StaleCompletionError,
)
from nari_qwen3_tts.executor.cache import PendingKVPublication


@dataclass(frozen=True, slots=True)
class _ClaimedRow:
    batch_id: int
    physical_row: int
    request_id: str
    stage: SynthesisStage
    expected_version: int
    logical_step: int
    token: int


@dataclass(frozen=True, slots=True)
class _ClaimedBatch:
    batch: StageExecutionBatch
    rows: tuple[_ClaimedRow, ...]


@dataclass(frozen=True, slots=True)
class ClaimHandle:
    """Opaque Engine-private identity for one all-or-nothing decision claim."""

    handle_id: int
    decision_id: int
    batch_ids: tuple[int, ...]


class Committer:
    """Validate, claim, publish, and release request lanes without planner tokens."""

    def __init__(self, store: RequestStateStore) -> None:
        self.store = store
        self._next_handle = 1
        self._next_token = 1
        self._claims: dict[int, ClaimHandle] = {}
        self._decision_handles: dict[int, int] = {}
        self._batches: dict[int, _ClaimedBatch] = {}
        self._batch_handles: dict[int, int] = {}
        self._completed_batches: set[int] = set()

    @property
    def active_handles(self) -> tuple[ClaimHandle, ...]:
        return tuple(self._claims[handle_id] for handle_id in sorted(self._claims))

    def _next_claim_token(self) -> int:
        token = self._next_token
        self._next_token += 1
        return token

    @staticmethod
    def _refresh_codec_readiness(state: RequestState) -> None:
        readiness = next_codec_readiness(state.codec)
        if readiness is None:
            return
        state.codec.ready_compatibility = readiness.compatibility
        state.codec.phase = readiness.phase
        state.codec.execution_reserve_s = readiness.execution_reserve_s

    @staticmethod
    def _lane(state: RequestState, stage: SynthesisStage):
        return state.codec if stage is SynthesisStage.CODEC else state.generation

    @classmethod
    def _lane_claim(cls, state: RequestState, stage: SynthesisStage) -> tuple[int | None, int | None]:
        lane = cls._lane(state, stage)
        return lane.claim_token, lane.claim_batch_id

    @classmethod
    def _set_lane_claim(
        cls,
        state: RequestState,
        stage: SynthesisStage,
        token: int | None,
        batch_id: int | None,
    ) -> None:
        lane = cls._lane(state, stage)
        lane.claim_token = token
        lane.claim_batch_id = batch_id

    @staticmethod
    def _require_ready(state: RequestState, batch: StageExecutionBatch, row: StageBatchRow) -> None:
        if state.cancel_requested:
            raise EngineStateError("cancelled requests cannot dispatch")
        if batch.stage is SynthesisStage.CODEC:
            if state.codec.phase is not CodecPhase.READY:
                raise EngineStateError("Codec lane is not ready")
            if row.logical_step != state.codec.chunk_index:
                raise EngineStateError("Codec row logical step is not currently ready")
            if row.compatibility != state.codec.ready_compatibility:
                raise EngineStateError("Codec row compatibility is not currently ready")
            return
        expected = {
            GenerationPhase.TALKER_PREFILL: SynthesisStage.TALKER_PREFILL,
            GenerationPhase.CODE_PREDICTOR: SynthesisStage.CODE_PREDICTOR,
            GenerationPhase.TALKER_DECODE: SynthesisStage.TALKER_DECODE,
        }.get(state.generation.phase)
        if expected is not batch.stage:
            raise EngineStateError(
                f"generation phase {state.generation.phase.value} cannot dispatch {batch.stage.value}"
            )
        if row.logical_step != state.generation.generation_step:
            raise EngineStateError("generation row logical step is not currently ready")

    def claim(self, decision: ScheduleDecision) -> ClaimHandle:
        if decision.decision_id in self._decision_handles:
            raise EngineStateError("decision was already claimed")
        if any(
            batch.batch_id in self._batches or batch.batch_id in self._completed_batches
            for batch in decision.batches
        ):
            raise EngineStateError("decision batch was already claimed")
        logical_rows = sum(batch.logical_rows for batch in decision.batches)
        if self.store.in_flight_rows + logical_rows > self.store.max_in_flight_rows:
            raise ResourceExhaustedError("request row claim pool is exhausted")

        claimed_batches: list[_ClaimedBatch] = []
        claimed_lanes: set[tuple[str, object]] = set()
        for batch in decision.batches:
            rows: list[_ClaimedRow] = []
            for row in batch.real_rows:
                assert row.request_id is not None and row.version is not None
                state = self.store.request(row.request_id)
                if state.version_for(batch.stage) != row.version:
                    raise StaleCompletionError("stage batch request version is stale")
                self._require_ready(state, batch, row)
                token, claimed_batch_id = self._lane_claim(state, batch.stage)
                lane_identity = (state.request_id, batch.stage.lane)
                if token is not None or claimed_batch_id is not None or lane_identity in claimed_lanes:
                    raise EngineStateError("request lane already has an in-flight claim")
                claimed_lanes.add(lane_identity)
                rows.append(
                    _ClaimedRow(
                        batch_id=batch.batch_id,
                        physical_row=row.physical_row,
                        request_id=row.request_id,
                        stage=batch.stage,
                        expected_version=row.version,
                        logical_step=row.logical_step,
                        token=self._next_claim_token(),
                    )
                )
            claimed_batches.append(_ClaimedBatch(batch, tuple(rows)))

        handle = ClaimHandle(
            handle_id=self._next_handle,
            decision_id=decision.decision_id,
            batch_ids=tuple(batch.batch_id for batch in decision.batches),
        )
        self._next_handle += 1
        for claimed in claimed_batches:
            for row in claimed.rows:
                state = self.store.request(row.request_id)
                self._set_lane_claim(state, row.stage, row.token, row.batch_id)
                if row.stage is SynthesisStage.CODEC:
                    state.codec.in_flight_compatibility = state.codec.ready_compatibility
                    state.codec.ready_compatibility = None
            self._batches[claimed.batch.batch_id] = claimed
            self._batch_handles[claimed.batch.batch_id] = handle.handle_id
        self.store._in_flight_rows += logical_rows
        self._claims[handle.handle_id] = handle
        self._decision_handles[decision.decision_id] = handle.handle_id
        return handle

    def batches(self, handle: ClaimHandle) -> tuple[StageExecutionBatch, ...]:
        self._require_handle(handle)
        return tuple(self._batches[batch_id].batch for batch_id in handle.batch_ids)

    def _require_handle(self, handle: ClaimHandle) -> None:
        if self._claims.get(handle.handle_id) != handle:
            raise EngineStateError("claim handle is stale or inactive")

    def _claimed_batch(self, handle: ClaimHandle, batch_id: int) -> _ClaimedBatch:
        self._require_handle(handle)
        if batch_id not in handle.batch_ids:
            raise EngineStateError("batch does not belong to the claim handle")
        claimed = self._batches.get(batch_id)
        if claimed is None:
            if batch_id in self._completed_batches:
                raise DuplicateCompletionError("stage completion was already consumed")
            raise EngineStateError("claim batch is stale or inactive")
        if self._batch_handles.get(batch_id) != handle.handle_id:
            raise EngineStateError("claim batch belongs to another handle")
        return claimed

    @staticmethod
    def _discard_publications(publications: tuple[PendingKVPublication, ...]) -> None:
        for publication in publications:
            try:
                publication.discard()
            except Exception:
                continue

    def _release_batch(self, handle: ClaimHandle, batch_id: int, *, restore: bool) -> None:
        claimed = self._claimed_batch(handle, batch_id)
        for row in claimed.rows:
            state = self.store.request(row.request_id)
            token, claimed_batch_id = self._lane_claim(state, row.stage)
            if token == row.token and claimed_batch_id == row.batch_id:
                self._set_lane_claim(state, row.stage, None, None)
                if state.cancel_requested:
                    if row.stage is SynthesisStage.CODEC:
                        state.codec.in_flight_compatibility = None
                        state.codec.ready_compatibility = None
                        state.codec.phase = CodecPhase.DONE
                    else:
                        state.generation.phase = GenerationPhase.DONE
                elif restore and row.stage is SynthesisStage.CODEC:
                    if state.codec.in_flight_compatibility is not None:
                        state.codec.ready_compatibility = state.codec.in_flight_compatibility
                        state.codec.in_flight_compatibility = None
                        state.codec.phase = CodecPhase.READY
                    else:
                        self._refresh_codec_readiness(state)
                elif row.stage is SynthesisStage.CODEC:
                    # Codec readiness cannot advance while its claim is held.
                    # Re-evaluate only after clearing it so a concurrently
                    # completed generation lane can expose partial or empty
                    # terminal work immediately.
                    self._refresh_codec_readiness(state)
        self.store._in_flight_rows -= len(claimed.rows)
        del self._batches[batch_id]
        del self._batch_handles[batch_id]
        self._completed_batches.add(batch_id)
        if all(batch not in self._batches for batch in handle.batch_ids):
            self._completed_batches.difference_update(handle.batch_ids)
            del self._claims[handle.handle_id]
            del self._decision_handles[handle.decision_id]

    @staticmethod
    def _validate_talker(
        state: RequestState,
        delta: TalkerStateDelta,
        publication: PendingKVPublication,
    ) -> None:
        if delta.kv is None or delta.kv.request_id != state.request_id:
            raise StaleCompletionError("Talker KV proposal belongs to another request")
        if getattr(publication, "request_id", state.request_id) != state.request_id:
            raise StaleCompletionError("Talker KV publication belongs to another request")
        validate = getattr(publication, "validate", None)
        if not callable(validate):
            raise StaleCompletionError("Talker KV publication lacks a validation boundary")
        expected_offset = state.generation.next_sampling_offset + TALKER_RNG_FRAME_STRIDE
        if delta.next_sampling_offset != expected_offset:
            raise StaleCompletionError("Talker sampling offset does not match the next RNG address")
        validate()

    @staticmethod
    def _validate_code_predictor(state: RequestState, delta: CodePredictorStateDelta) -> None:
        hidden = state.generation.hidden
        if hidden is None or tuple(delta.codec_sum.shape) != tuple(hidden.shape):
            raise StaleCompletionError("Code Predictor codec_sum shape must match Talker hidden state")

    @staticmethod
    def _validate_codec(state: RequestState, delta: CodecStateDelta) -> None:
        compatibility = state.codec.in_flight_compatibility
        if compatibility is None:
            raise StaleCompletionError("Codec completion has no claimed lifecycle plan")
        if (
            delta.consumed_frames != compatibility.producer_frames
            or delta.visible_frames != compatibility.visible_frames
            or delta.terminal is not compatibility.terminal
        ):
            raise StaleCompletionError("Codec completion does not match its claimed lifecycle plan")
        if (
            compatibility.mode in {CodecExecutionMode.COLD, CodecExecutionMode.WARM}
            and delta.state is None
        ):
            raise StaleCompletionError("incremental Codec completion requires successor state")

    @staticmethod
    def _apply_talker(
        state: RequestState,
        stage: SynthesisStage,
        delta: TalkerStateDelta,
        *,
        terminal: bool,
    ) -> None:
        state.generation.token = delta.token
        state.generation.hidden = delta.hidden
        state.generation.logits = delta.logits
        state.generation.seen_token_mask = delta.next_seen_token_mask
        state.generation.next_sampling_offset = delta.next_sampling_offset
        if stage is SynthesisStage.TALKER_PREFILL:
            state.generation.frame_index = 0
            state.generation.phase = GenerationPhase.CODE_PREDICTOR
        else:
            state.generation.generation_step += 1
            state.generation.frame_index += 1
            state.generation.phase = (
                GenerationPhase.DONE if terminal else GenerationPhase.CODE_PREDICTOR
            )
        state.generation.version += 1
        if state.generation.phase is GenerationPhase.DONE:
            state.codec.producer_done = True
            Committer._refresh_codec_readiness(state)

    @staticmethod
    def _apply_code_predictor(state: RequestState, delta: CodePredictorStateDelta) -> None:
        state.generation.step_input = delta.codec_sum
        state.generation.phase = GenerationPhase.TALKER_DECODE
        state.codec.buffered_frames = (*state.codec.buffered_frames, delta.frame)
        if state.codec.decoder_state is None:
            state.codec.history_frames = (*state.codec.history_frames, delta.frame)
        state.generation.version += 1
        Committer._refresh_codec_readiness(state)

    @staticmethod
    def _apply_codec(state: RequestState, delta: CodecStateDelta) -> None:
        compatibility = state.codec.in_flight_compatibility
        assert isinstance(compatibility, CodecBatchCompatibility)
        consumed = compatibility.producer_frames
        state.codec.buffered_frames = state.codec.buffered_frames[consumed:]
        state.codec.context_frames_consumed += consumed
        state.codec.visible_pcm_frames += compatibility.visible_frames
        if compatibility.mode in {CodecExecutionMode.COLD, CodecExecutionMode.WARM}:
            state.codec.decoder_state = delta.state
            if compatibility.mode is CodecExecutionMode.COLD:
                state.codec.history_frames = ()
        state.codec.chunk_index += 1
        state.codec.version += 1
        state.codec.in_flight_compatibility = None
        if compatibility.terminal:
            state.codec.compute_terminal = True
            state.codec.phase = CodecPhase.DONE
        else:
            state.codec.phase = CodecPhase.COLLECTING
            Committer._refresh_codec_readiness(state)

    def apply(  # noqa: PLR0912, PLR0915 - atomic completion transaction
        self,
        handle: ClaimHandle,
        completion: StageExecutionCompletion,
        *,
        kv_publications: tuple[PendingKVPublication, ...] = (),
        talker_terminal: tuple[bool, ...] = (),
    ) -> tuple[str, ...]:
        claimed = self._claimed_batch(handle, completion.batch_id)
        batch = claimed.batch
        if completion.stage is not batch.stage:
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            raise StaleCompletionError("completion stage does not match its claimed batch")
        if completion.error is not None:
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            return ()
        if tuple(result.row for result in completion.rows) != batch.rows:
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            raise StaleCompletionError("completion does not retain the claimed physical rows")

        results = tuple(result for result in completion.rows if not result.row.padding)
        if len(results) != len(claimed.rows):
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            raise StaleCompletionError("completion row count does not match its claimed batch")
        if batch.stage in {SynthesisStage.TALKER_PREFILL, SynthesisStage.TALKER_DECODE}:
            if len(kv_publications) != len(results) or len(talker_terminal) != len(results):
                self._discard_publications(kv_publications)
                self._release_batch(handle, batch.batch_id, restore=True)
                raise StaleCompletionError("Talker completion resources do not match its rows")
        elif kv_publications or talker_terminal:
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            raise StaleCompletionError("non-Talker completion cannot carry Talker resources")

        if any(self.store.request(row.request_id).cancel_requested for row in claimed.rows):
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            return ()

        states: list[RequestState] = []
        try:
            for index, (result, claim) in enumerate(zip(results, claimed.rows, strict=True)):
                state = self.store.request(claim.request_id)
                token, claimed_batch_id = self._lane_claim(state, claim.stage)
                if (
                    result.row.physical_row != claim.physical_row
                    or result.row.request_id != claim.request_id
                    or result.row.version != claim.expected_version
                    or result.row.logical_step != claim.logical_step
                    or state.version_for(claim.stage) != claim.expected_version
                    or token != claim.token
                    or claimed_batch_id != claim.batch_id
                ):
                    raise StaleCompletionError("completion row version or claim token is stale")
                delta = result.delta
                if isinstance(delta, TalkerStateDelta):
                    self._validate_talker(state, delta, kv_publications[index])
                elif isinstance(delta, CodePredictorStateDelta):
                    self._validate_code_predictor(state, delta)
                elif isinstance(delta, CodecStateDelta):
                    self._validate_codec(state, delta)
                else:
                    raise StaleCompletionError("completion carries the wrong typed state delta")
                states.append(state)
        except Exception:
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            raise

        try:
            for publication in reversed(kv_publications):
                publication.publish()
        except Exception:
            self._discard_publications(kv_publications)
            self._release_batch(handle, batch.batch_id, restore=True)
            raise

        committed: list[str] = []
        for index, (result, state) in enumerate(zip(results, states, strict=True)):
            delta = result.delta
            if isinstance(delta, TalkerStateDelta):
                self._apply_talker(
                    state,
                    batch.stage,
                    delta,
                    terminal=talker_terminal[index],
                )
            elif isinstance(delta, CodePredictorStateDelta):
                self._apply_code_predictor(state, delta)
            else:
                assert isinstance(delta, CodecStateDelta)
                self._apply_codec(state, delta)
            committed.append(state.request_id)
        self._release_batch(handle, batch.batch_id, restore=False)
        return tuple(committed)

    def reject(
        self,
        handle: ClaimHandle,
        *,
        batch_id: int,
        error: BaseException,
        kv_publications: tuple[PendingKVPublication, ...] = (),
    ) -> tuple[str, ...]:
        claimed = self._claimed_batch(handle, batch_id)
        completion = StageExecutionCompletion(
            batch_id=batch_id,
            stage=claimed.batch.stage,
            rows=(),
            error=error,
        )
        return self.apply(handle, completion, kv_publications=kv_publications)


__all__ = ["ClaimHandle", "Committer"]

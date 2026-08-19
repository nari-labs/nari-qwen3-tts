"""Engine-internal synthesis state machine and asynchronous GPU pipeline."""

from __future__ import annotations

import math
from concurrent.futures import Future
from dataclasses import dataclass, fields, replace

import torch

from nari_qwen3_tts.contract.errors import LiveInputClosedError, RequestCancelled
from nari_qwen3_tts.contract.model import SynthesisModelSpec
from nari_qwen3_tts.contract.request import AdmittedRequest
from nari_qwen3_tts.contract.result import (
    CodecStateDelta,
    CodePredictorStateDelta,
    KVPublication,
    StageBatchRowResult,
    StageExecutionCompletion,
    TalkerStateDelta,
)
from nari_qwen3_tts.contract.stage import SynthesisStage
from nari_qwen3_tts.contract.work import (
    CodecBatchCompatibility,
    CodePredictorBatchCompatibility,
    ReadyStageWork,
    ScheduleDecision,
    StageBatchRow,
    StageExecutionBatch,
    TalkerDecodeBatchCompatibility,
    TalkerPrefillBatchCompatibility,
)
from nari_qwen3_tts.engine.commit import ClaimHandle, Committer
from nari_qwen3_tts.engine.input_builder import InputBuilder
from nari_qwen3_tts.engine.output import OutputQueue
from nari_qwen3_tts.engine.state import (
    TALKER_RNG_FRAME_STRIDE,
    GenerationPhase,
    PendingLiveInput,
    RequestState,
    RequestStateStore,
)
from nari_qwen3_tts.engine.trace import TraceRecorder
from nari_qwen3_tts.executor.cache import PendingKVPublication
from nari_qwen3_tts.executor.rows import TalkerExecutionResult
from nari_qwen3_tts.executor.submission import (
    StageExecutionInputs,
    SubmissionFenceError,
    SubmissionWindow,
)
from nari_qwen3_tts.executor.types import CodecResult, CodePredictorResult
from nari_qwen3_tts.planner.catalog import CaptureCatalog
from nari_qwen3_tts.planner.planner import Planner
from nari_qwen3_tts.planner.policy import (
    DeadlineAwarePolicy,
    RoundRobinPolicy,
    SchedulingPolicy,
)
from nari_qwen3_tts.profile import RequiredSchedulingPolicy, SchedulingPolicyConfig

MAX_PENDING_LIVE_TEXT_TOKENS = 8_192

_TRACE_COMPATIBILITY_SCHEMAS = {
    compatibility_type: (
        compatibility_type.__name__,
        tuple(field.name for field in fields(compatibility_type)),
    )
    for compatibility_type in (
        TalkerPrefillBatchCompatibility,
        CodePredictorBatchCompatibility,
        TalkerDecodeBatchCompatibility,
        CodecBatchCompatibility,
    )
}
_TRACE_CLEANUP_FIELDS = ("request_id", "state_released", "execution_released")
_TRACE_OUTPUT_ROUTED_FIELDS = (
    "request_id",
    "pcm_bytes",
    "routed_at_s",
    "playback_started_at_s",
    "emitted_duration_s",
)
_TRACE_COMPLETION_SUCCESS_FIELDS = (
    "decision_id",
    "plan_id",
    "stage",
    "succeeded",
    "ticket_id",
)
_TRACE_COMPLETION_FAILURE_FIELDS = (
    "decision_id",
    "plan_id",
    "stage",
    "succeeded",
    "error",
)
_TRACE_COMMIT_FIELDS = (
    "decision_id",
    "plan_id",
    "ticket_id",
    "stage",
    "published",
    "rejected",
)


@dataclass(frozen=True, slots=True)
class StepResult:
    decision: ScheduleDecision
    completions: tuple[StageExecutionCompletion, ...]
    committed_request_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedExecution:
    """Engine-private, validated view of one immutable Planner decision."""

    decision: ScheduleDecision
    inputs: tuple[StageExecutionInputs, ...]
    batch_states: tuple[tuple[StageExecutionBatch, tuple[RequestState, ...]], ...]


@dataclass(frozen=True, slots=True)
class _CompletionEnvelope:
    completion: StageExecutionCompletion
    kv_publications: tuple[PendingKVPublication, ...] = ()
    talker_terminal: tuple[bool, ...] = ()
    codec_outputs: tuple[tuple[str, torch.Tensor, bool], ...] = ()


class SynthesisPipeline:
    """Own admitted-request state, planning, GPU submission, and completion."""

    def __init__(
        self,
        *,
        executor,
        capture_catalog: CaptureCatalog,
        policy_config: SchedulingPolicyConfig,
        model_config: SynthesisModelSpec,
        policy: SchedulingPolicy | None = None,
        max_in_flight_rows: int = 1024,
        trace_enabled: bool = False,
    ) -> None:
        self.executor = executor
        self.policy_config = policy_config
        self.model_config = model_config
        self.input_builder = InputBuilder(model_spec=model_config)
        self.state_store = RequestStateStore(max_in_flight_rows=max_in_flight_rows)
        self.catalog = capture_catalog
        if policy is None:
            if policy_config.kind is RequiredSchedulingPolicy.DEADLINE_AWARE:
                assert policy_config.pressing_lead_s is not None
                policy = DeadlineAwarePolicy(lead_s=policy_config.pressing_lead_s)
            else:
                policy = RoundRobinPolicy()
        if policy.name != policy_config.kind.value:
            raise ValueError(
                "injected scheduling policy does not match the profile-required policy "
                f"{policy_config.kind.value!r}"
            )
        if not isinstance(capture_catalog, CaptureCatalog):
            raise TypeError("SynthesisPipeline requires the canonical Planner capture catalog")
        self.planner = Planner(catalog=capture_catalog, policy=policy)
        self.committer = Committer(self.state_store)
        self._decision_sequence = 0
        self.trace = TraceRecorder(enabled=trace_enabled)
        self._admission_sequence = 0
        self._current_now_s = 0.0
        self.completion_event_factory = None
        self._submissions = SubmissionWindow(max_decisions=2)
        self._host_submission_contexts: dict[
            int,
            tuple[StageExecutionBatch, tuple[RequestState, ...], ClaimHandle],
        ] = {}
        self._overlap_dispatched: set[int] = set()
        self.output_queue: OutputQueue | None = None

    def attach_output_queue(self, output_queue: OutputQueue) -> None:
        if self.output_queue is not None:
            raise RuntimeError("Engine output queue is already attached")
        self.output_queue = output_queue

    def _throttle_cuda_submissions(self) -> None:
        self._submissions.reap_fences()
        if not self._submissions.can_submit:
            self._submissions.wait_oldest()

    def admit(self, admitted_request: AdmittedRequest) -> None:
        try:
            self.state_store.request(admitted_request.request_id)
        except KeyError:
            pass
        else:
            raise ValueError(f"request {admitted_request.request_id!r} is already admitted")
        admission_sequence = self._admission_sequence + 1
        state = RequestState(
            request_id=admitted_request.request_id,
            admission_sequence=admission_sequence,
            input=admitted_request,
        )
        state.codec.chunk_schedule = admitted_request.chunk_schedule
        state.codec.terminal_pad_frames = self.catalog.terminal_pad_frames
        state.codec.cold_terminal_pad_frames = self.catalog.cold_terminal_pad_frames
        state.codec.suppress_first_silent_frame = admitted_request.suppress_first_silent_frame
        state.codec.output_tracking = self.output_queue is not None
        self.executor.add_request(admitted_request.request_id)
        try:
            self.state_store.admit(state)
        except Exception:
            self.executor.remove_request(admitted_request.request_id)
            raise
        self._admission_sequence = admission_sequence

    def request(self, request_id: str) -> RequestState:
        return self.state_store.request(request_id)

    def cancel(self, request_id: str) -> None:
        state = self.request(request_id)
        self._fail_pending_request_input(
            state,
            RequestCancelled("live input was cancelled before publication"),
        )
        self.state_store.cancel(request_id)

    def remove(self, request_id: str) -> None:
        state = self.state_store.request(request_id)
        if not state.is_removable:
            raise RuntimeError("request still owns live or in-flight state")
        self.executor.remove_request(request_id)
        self.state_store.remove(request_id)
        if self.trace.enabled:
            self.trace.record_packed_fields(
                "cleanup",
                _TRACE_CLEANUP_FIELDS,
                (request_id, True, True),
            )

    def mark_pcm_routed(
        self,
        request_id: str,
        *,
        pcm_bytes: int,
        routed_at_s: float,
    ) -> None:
        """Publish playback credit only after PCM crosses the serving route boundary."""
        if isinstance(pcm_bytes, bool) or not isinstance(pcm_bytes, int):
            raise TypeError("routed PCM byte count must be an integer")
        if pcm_bytes < 0 or pcm_bytes % 2:
            raise ValueError("routed PCM must contain whole PCM16 samples")
        if isinstance(routed_at_s, bool) or not isinstance(routed_at_s, (int, float)):
            raise TypeError("PCM route time must be a number")
        if not math.isfinite(routed_at_s):
            raise ValueError("PCM route time must be finite")
        state = self.request(request_id)
        routed_at_s = float(routed_at_s)
        if (
            state.codec.last_routed_at_s is not None
            and routed_at_s < state.codec.last_routed_at_s
        ):
            raise ValueError("PCM route time must be monotonic")

        state.codec.last_routed_at_s = routed_at_s
        if pcm_bytes > 0:
            if state.codec.playback_started_at_s is None:
                state.codec.playback_started_at_s = routed_at_s
            state.codec.emitted_duration_s += (pcm_bytes // 2) / self.model_config.sample_rate
        if self.trace.enabled:
            self.trace.record_packed_fields(
                "output_routed",
                _TRACE_OUTPUT_ROUTED_FIELDS,
                (
                    request_id,
                    pcm_bytes,
                    routed_at_s,
                    state.codec.playback_started_at_s,
                    state.codec.emitted_duration_s,
                ),
            )

    def complete_pcm_output(
        self,
        request_id: str,
        *,
        pcm_bytes: int,
        routed_at_s: float | None,
        terminal_after: bool,
    ) -> None:
        """Release one materialized output and publish playback credit if routed."""

        state = self.request(request_id)
        if not state.codec.output_tracking or state.codec.pending_outputs < 1:
            raise RuntimeError("request does not own a pending PCM output")
        if isinstance(pcm_bytes, bool) or not isinstance(pcm_bytes, int):
            raise TypeError("PCM delivery byte count must be an integer")
        if pcm_bytes < 0 or pcm_bytes % 2:
            raise ValueError("PCM delivery must contain whole PCM16 samples")
        if routed_at_s is not None and pcm_bytes:
            self.mark_pcm_routed(
                request_id,
                pcm_bytes=pcm_bytes,
                routed_at_s=routed_at_s,
            )
        state.codec.pending_outputs -= 1
        if terminal_after:
            if not state.codec.compute_terminal:
                raise RuntimeError("output terminal preceded the Codec terminal commit")
            if state.codec.pending_outputs == 0:
                state.codec.output_terminal = True

    @staticmethod
    def _compatibility_trace(value: object) -> tuple[object, ...]:
        schema = _TRACE_COMPATIBILITY_SCHEMAS[type(value)]
        return schema, tuple(getattr(value, name) for name in schema[1])

    @classmethod
    def _work_trace(cls, work: ReadyStageWork) -> tuple[object, ...]:
        return (
            work.request_id,
            work.version,
            work.lane,
            work.stage,
            work.logical_step,
            work.admission_sequence,
            work.ready_sequence,
            work.startup,
            work.deadline_s,
            work.reserve_s,
            cls._compatibility_trace(work.compatibility),
        )

    @classmethod
    def _row_trace(cls, row: StageBatchRow, stage: SynthesisStage) -> tuple[object, ...]:
        return (
            row.physical_row,
            row.request_id,
            row.version,
            stage.lane,
            stage,
            row.logical_step,
            cls._compatibility_trace(row.compatibility),
            row.padding,
        )

    def update_request_input(
        self,
        request_id: str,
        token_ids: torch.Tensor,
        *,
        sequence: int,
        is_final: bool,
    ) -> Future[None]:
        return self.update_request_input_batch(
            request_id,
            ((token_ids, sequence, is_final),),
        )

    def update_request_input_batch(
        self,
        request_id: str,
        updates: tuple[tuple[torch.Tensor, int, bool], ...],
    ) -> Future[None]:
        return self._update_request_input_batch(
            request_id,
            updates,
            validate_token_values=True,
        )

    def update_validated_request_input_batch(
        self,
        request_id: str,
        updates: tuple[tuple[torch.Tensor, int, bool], ...],
    ) -> Future[None]:
        """Publish CPU-validated token IDs without synchronizing CUDA values."""
        return self._update_request_input_batch(
            request_id,
            updates,
            validate_token_values=False,
        )

    def _update_request_input_batch(
        self,
        request_id: str,
        updates: tuple[tuple[torch.Tensor, int, bool], ...],
        *,
        validate_token_values: bool,
    ) -> Future[None]:
        """Validate a live-input action completely before one state publication."""
        if not updates:
            raise ValueError("live input update batch cannot be empty")
        state = self.request(request_id)
        if state.cancel_requested or state.is_removable:
            raise RuntimeError("terminal requests cannot accept live input")
        if state.generation.phase is GenerationPhase.DONE:
            raise RuntimeError("finished generation cannot accept live input")
        admitted_input = self._admitted_input(state)
        staged = state.pending_live_input
        continuation = (
            admitted_input.talker_input.continuation
            if staged is None
            else staged.continuation
        )
        for token_ids, sequence, is_final in updates:
            if not isinstance(token_ids, torch.Tensor):
                raise TypeError("live input tokens must be tensors")
            if validate_token_values and token_ids.numel() and bool(
                torch.any(
                    (token_ids < 0) | (token_ids >= self.model_config.text_vocab_size)
                ).item()
            ):
                raise ValueError("live input contains an invalid text token ID")
            continuation = continuation.append(
                token_ids,
                sequence=sequence,
                is_final=is_final,
            )
            pending = max(
                0,
                continuation.token_count - state.generation.generation_step,
            )
            if pending > MAX_PENDING_LIVE_TEXT_TOKENS:
                raise ValueError("live input exceeds the pending text-token capacity")
        receipt: Future[None] = Future()
        if state.generation.claim_token is not None or staged is not None:
            if staged is None:
                state.pending_live_input = PendingLiveInput(continuation, [receipt])
            else:
                staged.continuation = continuation
                staged.receipts.append(receipt)
            return receipt
        self._publish_request_input(state, continuation)
        receipt.set_result(None)
        return receipt

    @staticmethod
    def _publish_request_input(
        state: RequestState,
        continuation,
    ) -> None:
        admitted_input = SynthesisPipeline._admitted_input(state)
        state.input = replace(
            admitted_input,
            talker_input=replace(admitted_input.talker_input, continuation=continuation),
        )
        state.generation.version += 1

    @staticmethod
    def _fail_pending_request_input(
        state: RequestState,
        error: BaseException,
    ) -> None:
        pending = state.pending_live_input
        if pending is None:
            return
        state.pending_live_input = None
        for receipt in pending.receipts:
            if not receipt.done():
                receipt.set_exception(error)

    def _publish_pending_request_inputs(self) -> tuple[str, ...]:
        published: list[str] = []
        for state in self.state_store.requests:
            pending = state.pending_live_input
            if pending is None:
                continue
            if state.cancel_requested:
                self._fail_pending_request_input(
                    state,
                    RequestCancelled("live input was cancelled before publication"),
                )
                continue
            if state.generation.claim_token is not None:
                continue
            if state.generation.phase is GenerationPhase.DONE:
                self._fail_pending_request_input(
                    state,
                    LiveInputClosedError("generation finished before live input publication"),
                )
                continue
            if all(receipt.cancelled() for receipt in pending.receipts):
                state.pending_live_input = None
                continue
            self._publish_request_input(state, pending.continuation)
            state.pending_live_input = None
            for receipt in pending.receipts:
                if not receipt.done():
                    receipt.set_result(None)
            published.append(state.request_id)
        return tuple(published)

    @staticmethod
    def _admitted_input(state: RequestState) -> AdmittedRequest:
        if not isinstance(state.input, AdmittedRequest):
            raise RuntimeError("request lacks its immutable admitted input")
        return state.input

    def _next_decision_id(self) -> int:
        self._decision_sequence += 1
        return self._decision_sequence

    def _claim_decision(self, decision: ScheduleDecision) -> ClaimHandle:
        try:
            claim = self.committer.claim(decision)
        except Exception:
            self.planner.discarded(decision)
            raise
        self.planner.committed(decision)
        if self.committer.batches(claim) != decision.batches:
            raise RuntimeError("claim ledger did not retain the Planner decision")
        return claim

    def has_ready_work(self) -> bool:
        """Report executable work without advancing ready-snapshot observation state."""
        return bool(
            self.planner.candidates(
                self.state_store.requests,
                now_s=self._current_now_s,
            )
        ) or bool(self._host_submission_contexts) or any(
            state.pending_live_input is not None for state in self.state_store.requests
        )

    def _prepare_execution(self, decision: ScheduleDecision) -> _PreparedExecution:
        batch_states = tuple(
            (batch, self._states(batch))
            for batch in decision.batches
        )
        inputs = tuple(
            self.input_builder.build_batch(batch, states)
            for batch, states in batch_states
        )
        self.executor.preflight(decision, inputs)
        return _PreparedExecution(decision, inputs, batch_states)

    def _suppress_decode_eos(self, state: RequestState) -> bool:
        return self.input_builder.suppress_decode_eos(state)

    def _states(self, batch: StageExecutionBatch) -> tuple[RequestState, ...]:
        return tuple(self.request(request_id) for request_id in batch.request_ids)

    @staticmethod
    def _completion_rows(
        batch: StageExecutionBatch,
        deltas: tuple[TalkerStateDelta | CodePredictorStateDelta | CodecStateDelta, ...],
    ) -> tuple[StageBatchRowResult, ...]:
        values = iter(deltas)
        rows = tuple(
            StageBatchRowResult(row, None if row.padding else next(values))
            for row in batch.rows
        )
        try:
            next(values)
        except StopIteration:
            return rows
        raise ValueError("stage result contains more logical rows than its batch")

    def _talker_completion(
        self,
        batch: StageExecutionBatch,
        output: TalkerExecutionResult,
        states: tuple[RequestState, ...],
        *,
        decode: bool,
    ) -> _CompletionEnvelope:
        tokens = output.result.tokens
        last_hidden = output.result.last_hidden
        logits = output.result.logits
        next_seen_token_masks = output.next_seen_token_masks
        kv_publications = output.kv_publications
        expected_rows = batch.logical_rows
        shaped = {
            "tokens": tokens,
            "last_hidden": last_hidden,
            "logits": logits,
            "next_seen_token_masks": next_seen_token_masks,
        }
        for name, value in shaped.items():
            shape = getattr(value, "shape", ())
            if not shape or int(shape[0]) != expected_rows:
                raise ValueError(f"Talker output {name} row count does not match the batch")
        if len(kv_publications) != expected_rows or len(states) != expected_rows:
            raise ValueError("Talker output row count does not match the stage batch")
        deltas: list[TalkerStateDelta] = []
        terminal: list[bool] = []
        for index, state in enumerate(states):
            request = self._admitted_input(state).request
            suppress_eos = decode and self._suppress_decode_eos(state)
            reached_limit = decode and (
                state.generation.generation_step + 1 >= request.effective_max_output_tokens
            )
            eos = False
            if decode and not suppress_eos and not request.ignore_eos and not reached_limit:
                eos = int(tokens[index].item()) == self.model_config.codec_eos_token_id
            proposal = kv_publications[index].proposal()
            if not isinstance(proposal, KVPublication):
                raise TypeError("Talker KV publication returned an invalid state proposal")
            deltas.append(
                TalkerStateDelta(
                    token=tokens[index],
                    hidden=last_hidden[index],
                    logits=logits[index],
                    next_seen_token_mask=next_seen_token_masks[index],
                    next_sampling_offset=(
                        state.generation.next_sampling_offset
                        + TALKER_RNG_FRAME_STRIDE
                    ),
                    kv=proposal,
                )
            )
            terminal.append(eos or reached_limit)
        return _CompletionEnvelope(
            completion=StageExecutionCompletion(
                batch_id=batch.batch_id,
                stage=batch.stage,
                rows=self._completion_rows(batch, tuple(deltas)),
            ),
            kv_publications=tuple(kv_publications),
            talker_terminal=tuple(terminal),
        )

    def _code_predictor_completion(
        self,
        batch: StageExecutionBatch,
        output: CodePredictorResult,
        states: tuple[RequestState, ...],
    ) -> _CompletionEnvelope:
        if (
            len(states) != batch.logical_rows
            or int(output.frames.shape[0]) != batch.logical_rows
            or int(output.codec_sum.shape[0]) != batch.logical_rows
        ):
            raise ValueError("Code Predictor output row count does not match the stage batch")
        deltas = tuple(
            CodePredictorStateDelta(
                frame=output.frames[index],
                codec_sum=output.codec_sum[index],
            )
            for index in range(batch.logical_rows)
        )
        return _CompletionEnvelope(
            StageExecutionCompletion(
                batch_id=batch.batch_id,
                stage=batch.stage,
                rows=self._completion_rows(batch, deltas),
            )
        )

    def _codec_completion(
        self,
        batch: StageExecutionBatch,
        output: CodecResult,
    ) -> _CompletionEnvelope:
        compatibility = batch.compatibility
        assert isinstance(compatibility, CodecBatchCompatibility)
        if not getattr(output.pcm, "shape", ()) or int(output.pcm.shape[0]) != batch.logical_rows:
            raise ValueError("Codec output row count does not match the stage batch")
        if output.states is not None and len(output.states) != batch.logical_rows:
            raise ValueError("Codec successor-state row count does not match the row manifest")
        if output.pcm_lengths is not None and len(output.pcm_lengths) != batch.logical_rows:
            raise ValueError("Codec PCM length row count does not match the row manifest")
        terminal_rows = {
            row.compatibility.terminal
            for row in batch.real_rows
            if isinstance(row.compatibility, CodecBatchCompatibility)
        }
        if terminal_rows != {output.terminal}:
            raise ValueError("Codec output terminal lifecycle does not match the row manifest")
        pcm_rows: list[torch.Tensor] = []
        deltas: list[CodecStateDelta] = []
        for index, row in enumerate(batch.real_rows):
            row_compatibility = row.compatibility
            assert isinstance(row_compatibility, CodecBatchCompatibility)
            expected = row_compatibility.visible_frames * self.model_config.samples_per_frame
            actual = (
                int(output.pcm.shape[1])
                if output.pcm_lengths is None
                else output.pcm_lengths[index]
            )
            if actual != expected:
                raise ValueError("Codec PCM row length does not match the row manifest")
            pcm = output.pcm[index, :actual]
            if row_compatibility.mode.value == "empty" and int(pcm.numel()) != 0:
                raise ValueError("empty metadata Codec completion cannot publish PCM")
            pcm_rows.append(pcm)
            deltas.append(
                CodecStateDelta(
                    state=(None if output.states is None else output.states[index]),
                    consumed_frames=row_compatibility.producer_frames,
                    visible_frames=row_compatibility.visible_frames,
                    terminal=row_compatibility.terminal,
                )
            )
        return _CompletionEnvelope(
            completion=StageExecutionCompletion(
                batch_id=batch.batch_id,
                stage=batch.stage,
                rows=self._completion_rows(batch, tuple(deltas)),
            ),
            codec_outputs=tuple(
                (row.request_id, pcm_rows[index], row.compatibility.terminal)
                for index, row in enumerate(batch.real_rows)
                if row.request_id is not None
            ),
        )

    def _publish_completion(
        self,
        batch: StageExecutionBatch,
        claim: ClaimHandle,
        envelope: _CompletionEnvelope,
    ) -> tuple[str, ...]:
        completion = envelope.completion
        succeeded = completion.error is None
        if self.trace.enabled:
            if succeeded:
                self.trace.record_packed_fields(
                    "completion",
                    _TRACE_COMPLETION_SUCCESS_FIELDS,
                    (batch.decision_id, batch.batch_id, batch.stage, True, batch.batch_id),
                )
            else:
                self.trace.record_packed_fields(
                    "completion",
                    _TRACE_COMPLETION_FAILURE_FIELDS,
                    (batch.decision_id, batch.batch_id, batch.stage, False, str(completion.error)),
                )
        committed = self.committer.apply(
            claim,
            completion,
            kv_publications=envelope.kv_publications,
            talker_terminal=envelope.talker_terminal,
        )
        if self.output_queue is not None and batch.stage is SynthesisStage.CODEC and committed:
            committed_ids = set(committed)
            outputs = tuple(
                output
                for output in envelope.codec_outputs
                if output[0] in committed_ids
            )
            self.output_queue.enqueue_many(outputs)
            for request_id, _source, _terminal in outputs:
                self.request(request_id).codec.pending_outputs += 1
        if self.trace.enabled:
            self.trace.record_packed_fields(
                "commit",
                _TRACE_COMMIT_FIELDS,
                (
                    batch.decision_id,
                    batch.batch_id,
                    batch.batch_id,
                    batch.stage,
                    committed,
                    not succeeded or not bool(committed),
                ),
            )
        return committed

    def _reject_batch(
        self,
        batch: StageExecutionBatch,
        claim: ClaimHandle,
        error: BaseException,
    ) -> None:
        self._publish_completion(
            batch,
            claim,
            _CompletionEnvelope(
                StageExecutionCompletion(
                    batch_id=batch.batch_id,
                    stage=batch.stage,
                    rows=(),
                    error=error,
                )
            ),
        )

    def _drain_host_submissions(
        self,
        *,
        block: bool,
    ) -> tuple[tuple[StageExecutionCompletion, ...], tuple[str, ...]]:
        """Collect finished deferrals, waiting on at most one of them.

        A blocking drain spends its single wait on the first deferral that is
        not already done and polls the rest, so one slow event cannot serialize
        the whole queue inside one call.  Callers that need the queue fully
        drained loop until it is empty.
        """
        try:
            ready = self._submissions.poll_host_ready(block_oldest=block)
        except SubmissionFenceError as error:
            submission = error.submission
            batch_id = submission.batch.batch_id
            context = self._host_submission_contexts.pop(batch_id, None)
            self._overlap_dispatched.discard(batch_id)
            if context is not None:
                self._reject_batch(context[0], context[2], error)
            raise
        completions: list[StageExecutionCompletion] = []
        committed: list[str] = []
        for submission in ready:
            batch_id = submission.batch.batch_id
            batch, states, claim = self._host_submission_contexts.pop(batch_id)
            self._overlap_dispatched.discard(batch_id)
            try:
                envelope = self._submission_completion(batch, states, submission.result)
            except Exception as error:
                self._reject_batch(batch, claim, error)
                raise
            completions.append(envelope.completion)
            committed.extend(self._publish_completion(batch, claim, envelope))
        return tuple(completions), tuple(committed)

    def _submission_completion(
        self,
        batch: StageExecutionBatch,
        states: tuple[RequestState, ...],
        output: object,
    ) -> _CompletionEnvelope:
        if batch.stage in {SynthesisStage.TALKER_PREFILL, SynthesisStage.TALKER_DECODE}:
            if not isinstance(output, TalkerExecutionResult):
                raise RuntimeError("Talker submission returned the wrong typed result")
            return self._talker_completion(
                batch,
                output,
                states,
                decode=batch.stage is SynthesisStage.TALKER_DECODE,
            )
        if batch.stage is SynthesisStage.CODE_PREDICTOR:
            if not isinstance(output, CodePredictorResult):
                raise RuntimeError("Code Predictor submission returned the wrong typed result")
            return self._code_predictor_completion(batch, output, states)
        if not isinstance(output, CodecResult):
            raise RuntimeError("Codec submission returned the wrong typed result")
        return self._codec_completion(batch, output)

    def execute_decision(  # noqa: PLR0912 - fail-closed submission cleanup is centralized
        self,
        prepared: _PreparedExecution,
        claim: ClaimHandle,
    ) -> StepResult:
        if not self._submissions.can_submit:
            self._throttle_cuda_submissions()
        decision = prepared.decision
        inputs = prepared.inputs
        contexts = tuple(
            (batch, states, claim)
            for batch, states in prepared.batch_states
        )
        completions: list[StageExecutionCompletion] = []
        committed: list[str] = []
        for batch in decision.batches:
            if self.trace.enabled:
                self.trace.record_dispatch(
                    (
                        decision.decision_id,
                        batch.batch_id,
                        batch.stage,
                        batch.request_ids,
                        tuple(self._row_trace(row, batch.stage) for row in batch.rows),
                    )
                )
        if hasattr(self.executor, "completion_event_factory"):
            self.executor.completion_event_factory = self.completion_event_factory
        try:
            submissions = self.executor.submit_preflighted(decision, inputs)
        except Exception as error:
            for batch, _states, batch_claim in contexts:
                self._reject_batch(batch, batch_claim, error)
            raise
        if len(submissions) != len(contexts):
            raise RuntimeError("executor submission count does not match the decision")
        for submission, (batch, states, batch_claim) in zip(submissions, contexts, strict=True):
            output = submission.result
            if submission.requires_host_finalize:
                if not isinstance(output, TalkerExecutionResult):
                    raise RuntimeError("host-finalize submission must contain Talker output")
                self._host_submission_contexts[batch.batch_id] = (batch, states, batch_claim)
                continue
            try:
                envelope = self._submission_completion(batch, states, output)
            except Exception as error:
                self._reject_batch(batch, batch_claim, error)
                raise
            completions.append(envelope.completion)
            committed.extend(self._publish_completion(batch, batch_claim, envelope))
        self._submissions.record(
            decision_id=decision.decision_id,
            submissions=submissions,
        )
        return StepResult(decision, tuple(completions), tuple(committed))

    def step(self, *, now_s: float) -> StepResult | None:
        self._current_now_s = now_s
        prefetched_candidates = None
        if not self._host_submission_contexts and not any(
            state.pending_live_input is not None for state in self.state_store.requests
        ):
            prefetched_candidates = self.planner.candidates(
                self.state_store.requests,
                now_s=now_s,
            )
            if not prefetched_candidates:
                return None
        self._throttle_cuda_submissions()
        drained_completions: list[StageExecutionCompletion] = []
        drained_committed: list[str] = []
        completions, committed = self._drain_host_submissions(block=False)
        drained_completions.extend(completions)
        drained_committed.extend(committed)
        while self._overlap_dispatched & self._host_submission_contexts.keys():
            completions, committed = self._drain_host_submissions(block=True)
            drained_completions.extend(completions)
            drained_committed.extend(committed)
        self._publish_pending_request_inputs()
        candidates = prefetched_candidates or self.planner.candidates(
            self.state_store.requests,
            now_s=now_s,
        )
        while not candidates and self._host_submission_contexts:
            completions, committed = self._drain_host_submissions(block=True)
            drained_completions.extend(completions)
            drained_committed.extend(committed)
            self._publish_pending_request_inputs()
            candidates = self.planner.candidates(self.state_store.requests, now_s=now_s)
        if not candidates:
            return None
        canonical_decision = self.planner.plan(
            candidates,
            now_s=now_s,
            decision_id=self._next_decision_id(),
            available_rows=self.state_store.available_in_flight_rows,
            record_observation=self.trace.enabled,
        )
        observation = (
            self.planner.observation(canonical_decision)
            if self.trace.enabled
            else None
        )
        try:
            prepared = self._prepare_execution(canonical_decision)
        except Exception:
            self.planner.discarded(canonical_decision)
            raise
        claim = self._claim_decision(canonical_decision)
        if self.trace.enabled:
            assert observation is not None
            compatibility_partition = tuple(
                (
                    batch.stage,
                    self._compatibility_trace(batch.compatibility),
                    batch.request_ids,
                )
                for batch in canonical_decision.batches
            )
            self.trace.record_decision(
                (
                    canonical_decision.decision_id,
                    observation.snapshot_sequence,
                    tuple(self._work_trace(work) for work in observation.ready),
                    tuple(self._work_trace(work) for work in observation.eligible),
                    tuple(self._work_trace(work) for work in observation.selected),
                    compatibility_partition,
                    tuple(
                        (batch.logical_rows, len(batch.rows), batch.padding_rows)
                        for batch in canonical_decision.batches
                    ),
                    tuple(
                        (wait.request_id, wait.stage, wait.reason, wait.wait_decisions)
                        for wait in observation.wait_reasons
                    ),
                    observation.evaluation.reason,
                    observation.evaluation.policy_inputs,
                    observation.evaluation.rr_counterfactual,
                    tuple(
                        tuple(self._row_trace(row, batch.stage) for row in batch.rows)
                        for batch in canonical_decision.batches
                    ),
                )
            )
        existing_pending = set(self._host_submission_contexts)
        result = self.execute_decision(prepared, claim)
        self._overlap_dispatched.update(existing_pending)
        return StepResult(
            result.decision,
            (*drained_completions, *result.completions),
            (*drained_committed, *result.committed_request_ids),
        )

    def normalized_trace(self) -> tuple[dict[str, object], ...]:
        return self.trace.normalized()


__all__ = [
    "SynthesisPipeline",
    "MAX_PENDING_LIVE_TEXT_TOKENS",
    "StepResult",
]

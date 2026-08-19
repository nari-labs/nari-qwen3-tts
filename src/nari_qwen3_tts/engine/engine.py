"""Single-threaded synthesis Engine command and execution loop."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch

from nari_qwen3_tts.config import EngineConfig
from nari_qwen3_tts.contract.command import (
    AppendText,
    CancelRequest,
    EngineCommand,
    StopEngine,
    SubmitRequest,
)
from nari_qwen3_tts.contract.errors import (
    BackpressureExceeded,
    LiveInputClosedError,
    RequestCancelled,
    RequestRejected,
    ServiceCapacityExceeded,
    ServiceUnavailable,
    SynthesisError,
)
from nari_qwen3_tts.contract.health import (
    EngineHealth,
    ReadinessStatus,
    ServicePhase,
    evaluate_readiness,
)
from nari_qwen3_tts.contract.request import AdmittedRequest, SynthesisRequest
from nari_qwen3_tts.contract.stream import PCMStream
from nari_qwen3_tts.engine.admission import make_admitted_request
from nari_qwen3_tts.engine.output import OutputQueue, PcmDelivery
from nari_qwen3_tts.engine.pipeline import SynthesisPipeline
from nari_qwen3_tts.engine.state import LiveInputState
from nari_qwen3_tts.executor.executor import Executor
from nari_qwen3_tts.executor.pcm import PcmTransferPool
from nari_qwen3_tts.planner.catalog import CaptureCoverageError


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    admitted: int = 0
    completed: int = 0
    cancelled: int = 0
    backpressured: int = 0
    failed: int = 0
    active: int = 0

    def __post_init__(self) -> None:
        for name in ("admitted", "completed", "cancelled", "backpressured", "failed", "active"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"service metric {name} must be an integer")
            if value < 0:
                raise ValueError(f"service metric {name} cannot be negative")


@dataclass(slots=True)
class _ClientSession:
    request_id: str
    request: SynthesisRequest
    stream: PCMStream
    live_state: LiveInputState | None = None
    internal_probe: bool = False
    admitted: bool = False
    cancel_sent: bool = False
    cancel_accounted: bool = False
    ordinary_pcm_bytes: int = 0
    retired: bool = False


def _empty_health() -> EngineHealth:
    return EngineHealth(0, 0, 0, 0, 0, 0, ())


class Engine:
    """Own request lifecycle and concrete GPU execution on one Engine thread."""

    def __init__(
        self,
        model,
        *,
        executor: Executor,
        pipeline: SynthesisPipeline,
        capture_catalog,
        config: EngineConfig | None = None,
    ) -> None:
        self.model = model
        self.executor = executor
        self.pipeline = pipeline
        self.catalog = capture_catalog
        maximum_frames = max(
            key.model_frames
            for key in capture_catalog.codec
        )
        maximum_samples = maximum_frames * executor.codec.samples_per_frame
        self._pcm_pool = PcmTransferPool(
            maximum_samples=maximum_samples,
            device=torch.device(executor.talker.model.device),
        )
        engine_config = config or EngineConfig()
        self._pcm_pool.prepare(2 * engine_config.max_active_requests)
        self.output = OutputQueue(self._pcm_pool)
        self.pipeline.attach_output_queue(self.output)
        self._initialize_loop(engine_config)

    def _initialize_loop(self, config: EngineConfig) -> None:
        self.config = config
        self._commands: queue.Queue[EngineCommand] = queue.Queue(
            maxsize=self.config.command_capacity
        )
        self._orphan_cancellations: queue.SimpleQueue[CancelRequest] = queue.SimpleQueue()
        self._live_input_failures: queue.SimpleQueue[tuple[str, BaseException]] = queue.SimpleQueue()
        self._records: dict[str, _ClientSession] = {}
        self._thread: threading.Thread | None = None
        self._engine_thread_id: int | None = None
        self._condition = threading.Condition()
        self._phase = ServicePhase.STARTING
        self._failure: BaseException | None = None
        self._failed_after_ready = False
        self._last_health = _empty_health()
        self._ordinary_pcm_bytes = 0
        self._ordinary_stage_deltas: dict[str, int] = {}
        self._ordinary_retired = False
        self._metrics = EngineMetrics()
        self._stop_requested = False
        self._commands_in_flight = 0

    @staticmethod
    def _execution_context():
        return nullcontext()

    def _execution_health(self) -> EngineHealth:
        return self.executor.health()

    def _admit_request(
        self,
        request_id: str,
        request: SynthesisRequest,
        *,
        admitted_at_s: float,
        live: bool,
        initial_token_ids: tuple[int, ...] = (),
        initial_wrapped_ids: tuple[int, ...] = (),
        input_finished: bool = True,
    ) -> None:
        prepared = (
            self.model.prepare_live(
                request,
                token_ids=initial_token_ids,
                wrapped_ids=initial_wrapped_ids,
            )
            if live
            else self.model.prepare(request)
        )
        plan = self.executor.talker.prepare_prepared_inputs([prepared])
        admitted = make_admitted_request(
            request_id=request_id,
            request=prepared.request,
            talker_plan=plan,
            execution_config=self.executor.config,
            admitted_at_s=admitted_at_s,
            input_finished=input_finished,
        )
        try:
            self.catalog.lower_talker_prefill(plan.sequence_lengths)
            self.catalog.validate_codec_schedule(admitted.chunk_schedule)
        except CaptureCoverageError as error:
            raise RequestRejected(str(error)) from error
        self.pipeline.admit(admitted)

    def _update_request_input_batch(
        self,
        request_id: str,
        updates: tuple[tuple[tuple[int, ...], int, bool], ...],
    ) -> Future[None]:
        trace = self._diagnostic_trace()
        trace_enabled = trace is not None and trace.enabled
        started_ns = time.perf_counter_ns() if trace_enabled else 0
        state = self.pipeline.request(request_id)
        admitted_input = state.input
        if not isinstance(admitted_input, AdmittedRequest):
            raise RuntimeError("synthesis request lacks its typed admitted input")
        continuation = admitted_input.talker_input.continuation
        vocab_size = self.pipeline.model_config.text_vocab_size
        if any(
            type(token_id) is not int or not 0 <= token_id < vocab_size
            for token_ids, _sequence, _is_final in updates
            for token_id in token_ids
        ):
            raise ValueError("live input contains an invalid text token ID")
        tensors = tuple(
            (
                torch.tensor(
                    token_ids,
                    dtype=continuation.token_ids.dtype,
                    device="cpu",
                ),
                sequence,
                is_final,
            )
            for token_ids, sequence, is_final in updates
        )
        staged_ns = time.perf_counter_ns() if trace_enabled else 0
        receipt = self.pipeline.update_validated_request_input_batch(request_id, tensors)
        if trace_enabled:
            completed_ns = time.perf_counter_ns()
            trace.record_packed_fields(
                "engine_live_token_stage",
                (
                    "duration_ns",
                    "host_tensor_stage_ns",
                    "pipeline_update_ns",
                    "tokens",
                    "updates",
                ),
                (
                    completed_ns - started_ns,
                    staged_ns - started_ns,
                    completed_ns - staged_ns,
                    sum(len(token_ids) for token_ids, _sequence, _is_final in updates),
                    len(updates),
                ),
            )
        return receipt

    def _step_execution(self, *, now_s: float):
        return self.pipeline.step(now_s=now_s)

    def _cancel_request(self, request_id: str) -> None:
        self.output.cancel_request(request_id)
        self.pipeline.cancel(request_id)

    def _request_is_removable(self, request_id: str) -> bool:
        return self.pipeline.request(request_id).is_removable

    def _remove_request(self, request_id: str) -> None:
        discarded = self.output.discard_request(request_id)
        state = self.pipeline.request(request_id)
        state.codec.pending_outputs -= discarded
        if state.codec.pending_outputs < 0:
            raise RuntimeError("pending PCM output count underflowed")
        self.pipeline.remove(request_id)

    @property
    def engine_thread_id(self) -> int | None:
        return self._engine_thread_id

    def _assert_engine_thread(self) -> None:
        if self._thread is not None and threading.get_ident() != self._engine_thread_id:
            raise RuntimeError("Engine state may be accessed only on the Engine thread")

    def _diagnostic_trace(self):
        pipeline = getattr(self, "pipeline", None)
        return None if pipeline is None else pipeline.trace

    def metrics(self) -> EngineMetrics:
        with self._condition:
            return self._metrics

    def readiness(self) -> ReadinessStatus:
        with self._condition:
            return evaluate_readiness(
                phase=self._phase,
                health=self._last_health,
                ordinary_pcm_bytes=self._ordinary_pcm_bytes,
                ordinary_stage_deltas=dict(self._ordinary_stage_deltas),
                ordinary_retired=self._ordinary_retired,
                failed_after_ready=self._failed_after_ready,
            )

    def start(self, readiness_request: SynthesisRequest, *, timeout_s: float = 120.0) -> None:
        if not isinstance(readiness_request, SynthesisRequest):
            raise TypeError("readiness_request must be a SynthesisRequest")
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("serving service can start only once")
            self._thread = threading.Thread(
                target=self._run,
                args=(readiness_request,),
                name="nari-qwen3-tts-engine",
                daemon=True,
            )
            self._thread.start()
            deadline = time.monotonic() + timeout_s
            while self._phase is ServicePhase.STARTING:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_requested = True
                    raise TimeoutError("timed out waiting for ordinary TTS readiness proof")
                self._condition.wait(remaining)
            if self._phase is not ServicePhase.READY:
                error = ServiceUnavailable("Qwen3-TTS serving startup failed")
                if self._failure is not None:
                    raise error from self._failure
                raise error

    def _put(self, command: EngineCommand, *, timeout_s: float) -> None:
        try:
            self._commands.put(command, timeout=timeout_s)
        except queue.Full as error:
            raise ServiceCapacityExceeded("serving command queue capacity exceeded") from error

    def _require_ready(self) -> None:
        status = self.readiness()
        if not status.ready:
            raise ServiceUnavailable(f"Qwen3-TTS service is not ready: {status.reason}")

    def _wait_future(
        self,
        future: Future,
        *,
        timeout_s: float,
        request_id: str | None = None,
    ):
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError:
            cancelled_before_execution = future.cancel()
            if not cancelled_before_execution and request_id is not None:
                self._orphan_cancellations.put(CancelRequest(request_id, Future()))
            raise

    def submit(
        self,
        request_id: str,
        request: SynthesisRequest,
        *,
        timeout_s: float = 10.0,
    ) -> PCMStream:
        self._require_ready()
        future: Future[PCMStream] = Future()
        self._put(
            SubmitRequest(request_id, request, False, (), (), True, future),
            timeout_s=timeout_s,
        )
        return self._wait_future(future, timeout_s=timeout_s, request_id=request_id)

    def begin_live(
        self,
        request_id: str,
        request: SynthesisRequest,
        *,
        initial_token_ids: tuple[int, ...],
        initial_wrapped_ids: tuple[int, ...],
        input_finished: bool,
        timeout_s: float = 10.0,
    ) -> PCMStream:
        self._require_ready()
        if request.non_streaming_mode:
            request = replace(request, non_streaming_mode=False)
        future: Future[PCMStream] = Future()
        self._put(
            SubmitRequest(
                request_id,
                request,
                True,
                initial_token_ids,
                initial_wrapped_ids,
                input_finished,
                future,
            ),
            timeout_s=timeout_s,
        )
        return self._wait_future(future, timeout_s=timeout_s, request_id=request_id)

    def append_text(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        *,
        sequence: int,
        is_final: bool,
        timeout_s: float = 10.0,
    ) -> None:
        self._require_ready()
        future: Future[None] = Future()
        self._put(
            AppendText(request_id, sequence, token_ids, is_final, future),
            timeout_s=timeout_s,
        )
        self._wait_future(future, timeout_s=timeout_s, request_id=request_id)

    def cancel(self, request_id: str, *, timeout_s: float = 10.0) -> None:
        self._require_ready()
        future: Future[None] = Future()
        self._put(CancelRequest(request_id, future), timeout_s=timeout_s)
        self._wait_future(future, timeout_s=timeout_s)

    def wait_idle(self, *, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while (
                self._records
                or not self._commands.empty()
                or not self._orphan_cancellations.empty()
                or not self._live_input_failures.empty()
                or self._commands_in_flight
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def stop(self, *, timeout_s: float = 10.0) -> None:
        with self._condition:
            thread = self._thread
            if thread is None:
                self._phase = ServicePhase.STOPPED
                return
            if not thread.is_alive():
                self._phase = ServicePhase.STOPPED
                return
            starting = self._phase is ServicePhase.STARTING
            if starting:
                self._stop_requested = True
        if starting:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                raise TimeoutError("Engine thread did not stop")
            with self._condition:
                self._phase = ServicePhase.STOPPED
            return
        future: Future[None] = Future()
        self._put(StopEngine(future), timeout_s=timeout_s)
        self._wait_future(future, timeout_s=timeout_s)
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise TimeoutError("Engine thread did not stop")

    def _set_metrics(self, **changes: int) -> None:
        with self._condition:
            current = self._metrics
            values = {
                "admitted": current.admitted,
                "completed": current.completed,
                "cancelled": current.cancelled,
                "backpressured": current.backpressured,
                "failed": current.failed,
                "active": current.active,
            }
            for name, amount in changes.items():
                values[name] += amount
            self._metrics = EngineMetrics(**values)

    def _admit_record(
        self,
        record: _ClientSession,
        *,
        request: SynthesisRequest,
        live: bool,
        initial_token_ids: tuple[int, ...] = (),
        initial_wrapped_ids: tuple[int, ...] = (),
        input_finished: bool = True,
    ) -> None:
        self._assert_engine_thread()
        self._admit_request(
            record.request_id,
            request,
            admitted_at_s=time.monotonic(),
            live=live,
            initial_token_ids=initial_token_ids,
            initial_wrapped_ids=initial_wrapped_ids,
            input_finished=input_finished,
        )
        record.admitted = True

    def _handle_submit(self, command: SubmitRequest) -> None:
        if command.reply.cancelled():
            return
        if not isinstance(command.request_id, str) or not command.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(command.request, SynthesisRequest):
            raise TypeError("service request must be a SynthesisRequest")
        if command.request_id in self._records:
            raise ValueError(f"request {command.request_id!r} is already active")
        if len(self._records) >= self.config.max_active_requests:
            raise ServiceCapacityExceeded("active request capacity exceeded")
        record = _ClientSession(
            command.request_id,
            command.request,
            PCMStream(max_buffered_bytes=self.config.max_buffered_pcm_bytes),
            live_state=(
                LiveInputState(
                    next_engine_sequence=0,
                    input_finished=command.input_finished,
                    committed_tokens=len(command.initial_token_ids),
                )
                if command.live
                else None
            ),
        )
        self._admit_record(
            record,
            request=command.request,
            live=command.live,
            initial_token_ids=command.initial_token_ids,
            initial_wrapped_ids=command.initial_wrapped_ids,
            input_finished=command.input_finished,
        )
        if command.reply.cancelled():
            self._cancel_request(record.request_id)
            if self._request_is_removable(record.request_id):
                self._remove_request(record.request_id)
                return
            record.cancel_sent = True
        with self._condition:
            self._records[record.request_id] = record
        self._set_metrics(admitted=1, active=1)
        if command.reply.cancelled():
            self._request_cancel(record, RequestCancelled("submit command timed out"))
            return
        command.reply.set_result(record.stream)

    def _validate_append(self, command: AppendText) -> tuple[_ClientSession, LiveInputState, int]:
        try:
            record = self._records[command.request_id]
        except KeyError as error:
            raise KeyError(f"unknown live request {command.request_id!r}") from error
        live = record.live_state
        if live is None:
            raise RuntimeError("request does not own live text input")
        if live.input_finished:
            raise RuntimeError("live text input is already finished")
        if command.sequence != live.next_engine_sequence:
            raise ValueError(
                f"live input sequence {command.sequence!r} does not match "
                f"expected {live.next_engine_sequence}"
            )
        if len(command.token_ids) > self.config.live_input.max_update_tokens:
            raise ServiceCapacityExceeded("live text update token capacity exceeded")
        next_token_count = live.committed_tokens + len(command.token_ids)
        if next_token_count > self.config.live_input.max_live_text_tokens:
            raise ServiceCapacityExceeded("live text token capacity exceeded")
        return record, live, next_token_count

    def _observe_live_publication(
        self,
        request_id: str,
        receipt: Future[None],
    ) -> None:
        def completed(value: Future[None]) -> None:
            try:
                value.result()
            except BaseException as error:
                self._live_input_failures.put((request_id, error))

        receipt.add_done_callback(completed)

    def _handle_append(self, command: AppendText) -> None:
        if command.reply.cancelled():
            return
        trace = self._diagnostic_trace()
        trace_enabled = trace is not None and trace.enabled
        started_ns = time.perf_counter_ns() if trace_enabled else 0
        record, previous, next_token_count = self._validate_append(command)
        validated_ns = time.perf_counter_ns() if trace_enabled else 0
        receipt = self._update_request_input_batch(
            record.request_id,
            ((command.token_ids, command.sequence, command.is_final),),
        )
        self._observe_live_publication(record.request_id, receipt)
        record.live_state = LiveInputState(
            next_engine_sequence=previous.next_engine_sequence + 1,
            input_finished=command.is_final,
            committed_tokens=next_token_count,
        )
        if not command.reply.done():
            command.reply.set_result(None)
        if trace_enabled:
            trace.record_packed_fields(
                "engine_append_text",
                (
                    "duration_ns",
                    "validate_ns",
                    "queue_ns",
                    "token_count",
                    "publication_pending",
                ),
                (
                    time.perf_counter_ns() - started_ns,
                    validated_ns - started_ns,
                    time.perf_counter_ns() - validated_ns,
                    len(command.token_ids),
                    int(not receipt.done()),
                ),
            )

    def _request_cancel(self, record: _ClientSession, error: SynthesisError) -> None:
        if record.admitted and not record.cancel_sent:
            self._cancel_request(record.request_id)
            record.cancel_sent = True
        record.stream.fail(error)

    def _drain_live_input_failures(self) -> None:
        while True:
            try:
                request_id, error = self._live_input_failures.get_nowait()
            except queue.Empty:
                return
            record = self._records.get(request_id)
            if record is None or record.retired:
                continue
            failure = (
                error
                if isinstance(error, SynthesisError)
                else LiveInputClosedError(str(error) or "live input publication failed")
            )
            self._request_cancel(record, failure)

    def _handle_cancel(self, command: CancelRequest) -> None:
        if command.reply.cancelled():
            return
        try:
            record = self._records[command.request_id]
        except KeyError as error:
            raise KeyError(f"unknown request {command.request_id!r}") from error
        self._request_cancel(record, RequestCancelled("request was cancelled"))
        if not record.cancel_accounted:
            self._set_metrics(cancelled=1)
            record.cancel_accounted = True
        if not record.admitted:
            self._retire(record, completed=False)
        command.reply.set_result(None)

    def _handle_command(self, command: EngineCommand) -> None:
        if command.reply.cancelled():
            return
        try:
            if isinstance(command, SubmitRequest):
                self._handle_submit(command)
            elif isinstance(command, AppendText):
                self._handle_append(command)
            elif isinstance(command, CancelRequest):
                self._handle_cancel(command)
            else:
                self._stop_requested = True
                for record in tuple(self._records.values()):
                    self._request_cancel(record, RequestCancelled("service is stopping"))
                    if not record.admitted:
                        self._retire(record, completed=False)
                command.reply.set_result(None)
        except BaseException as error:
            if not command.reply.cancelled():
                command.reply.set_exception(error)

    def _execute_command(self, command: EngineCommand) -> None:
        self._assert_engine_thread()
        if not command.reply.set_running_or_notify_cancel():
            return
        with self._condition:
            self._commands_in_flight += 1
        try:
            self._handle_command(command)
        finally:
            with self._condition:
                self._commands_in_flight -= 1
                self._condition.notify_all()

    def _retire(self, record: _ClientSession, *, completed: bool) -> None:
        if record.retired:
            return
        if record.admitted:
            self._remove_request(record.request_id)
        record.retired = True
        with self._condition:
            self._records.pop(record.request_id, None)
        self._set_metrics(completed=int(completed), active=-1)
        with self._condition:
            self._condition.notify_all()

    def _deliver_pcm(self, delivery: PcmDelivery) -> None:
        record = self._records.get(delivery.request_id)
        if record is None:
            raise RuntimeError("materialized PCM belongs to an unknown request")
        routed_at_s = None
        try:
            if not delivery.discarded and not record.cancel_sent:
                if record.internal_probe:
                    record.ordinary_pcm_bytes += len(delivery.value)
                else:
                    record.stream.publish(delivery.value)
                    if delivery.value:
                        routed_at_s = time.monotonic()
            self.pipeline.complete_pcm_output(
                record.request_id,
                pcm_bytes=len(delivery.value),
                routed_at_s=routed_at_s,
                terminal_after=delivery.terminal_after,
            )
        except BackpressureExceeded:
            self.pipeline.complete_pcm_output(
                record.request_id,
                pcm_bytes=len(delivery.value),
                routed_at_s=None,
                terminal_after=delivery.terminal_after,
            )
            self._set_metrics(backpressured=1)
            self._request_cancel(
                record,
                BackpressureExceeded("request PCM backpressure budget exceeded"),
            )
            return
        except Exception as error:
            self._set_metrics(failed=1)
            self._request_cancel(
                record,
                ServiceUnavailable(f"request PCM delivery failed: {error}"),
            )
            return
        state = self.pipeline.request(record.request_id)
        if (
            delivery.terminal_after
            and state.codec.pending_outputs == 0
            and not record.internal_probe
            and not record.stream.terminal
        ):
            record.stream.close()

    def _poll_records(self) -> None:
        self._assert_engine_thread()
        trace = self._diagnostic_trace()
        trace_started_ns = time.perf_counter_ns() if trace is not None and trace.enabled else 0
        deliveries = self.output.poll_ready(request_order=tuple(self._records))
        for delivery in deliveries:
            self._deliver_pcm(delivery)
        retired = 0
        for record in tuple(self._records.values()):
            if not record.admitted:
                continue
            state = self.pipeline.request(record.request_id)
            if state.cancel_requested and not record.stream.terminal:
                record.stream.fail(RequestCancelled("request was cancelled"))
            if state.is_removable:
                if record.internal_probe:
                    self._ordinary_pcm_bytes = record.ordinary_pcm_bytes
                    self._ordinary_retired = True
                self._retire(
                    record,
                    completed=state.codec.output_terminal and not state.cancel_requested,
                )
                retired += 1
        if trace is not None and trace.enabled:
            trace.record_packed_fields(
                "engine_pcm_poll",
                ("duration_ns", "deliveries", "retired", "active_requests"),
                (
                    time.perf_counter_ns() - trace_started_ns,
                    len(deliveries),
                    retired,
                    len(self._records),
                ),
            )

    def _warmup(self, request: SynthesisRequest) -> None:
        self._assert_engine_thread()
        baseline = self._execution_health()
        self._last_health = baseline
        if not baseline.capture_ready:
            raise ServiceUnavailable("CUDA capture health is incomplete before readiness probe")
        request_id = f"readiness-{uuid.uuid4().hex}"
        record = _ClientSession(
            request_id,
            request,
            PCMStream(max_buffered_bytes=self.config.max_buffered_pcm_bytes),
            internal_probe=True,
        )
        self._admit_record(record, request=request, live=False)
        with self._condition:
            self._records[request_id] = record
        self._set_metrics(admitted=1, active=1)
        while request_id in self._records:
            if self._stop_requested:
                self._request_cancel(record, RequestCancelled("readiness probe was stopped"))
                self._poll_records()
                if request_id not in self._records:
                    raise ServiceUnavailable("readiness probe was cancelled")
            self._step_execution(now_s=time.monotonic())
            self._poll_records()
        after = self._execution_health()
        baseline_counts = baseline.submitted_by_stage()
        self._ordinary_stage_deltas = {
            name: submitted - baseline_counts.get(name, 0)
            for name, submitted in after.submitted_by_stage().items()
        }
        self._last_health = after
        verdict = evaluate_readiness(
            phase=ServicePhase.READY,
            health=after,
            ordinary_pcm_bytes=self._ordinary_pcm_bytes,
            ordinary_stage_deltas=self._ordinary_stage_deltas,
            ordinary_retired=self._ordinary_retired,
        )
        if not verdict.ready:
            raise ServiceUnavailable(f"ordinary TTS readiness probe failed: {verdict.reason}")

    def _next_command(self) -> EngineCommand | None:
        try:
            return self._orphan_cancellations.get_nowait()
        except queue.Empty:
            pass
        try:
            if self._records:
                return self._commands.get_nowait()
            return self._commands.get(timeout=self.config.idle_poll_interval_s)
        except queue.Empty:
            return None

    def _next_queued_command_nowait(self) -> EngineCommand:
        try:
            return self._orphan_cancellations.get_nowait()
        except queue.Empty:
            pass
        return self._commands.get_nowait()

    def _drain_commands(self, first: EngineCommand) -> None:
        trace = self._diagnostic_trace()
        if trace is None or not trace.enabled:
            self._execute_command(first)
            for _ in range(self.config.max_commands_per_turn - 1):
                try:
                    command = self._next_queued_command_nowait()
                except queue.Empty:
                    return
                self._execute_command(command)
            return

        trace_started_ns = time.perf_counter_ns()
        executed = 0
        appended = 0
        submitted = 0
        cancelled = 0

        def execute(command: EngineCommand) -> None:
            nonlocal executed, appended, submitted, cancelled
            executed += 1
            appended += int(isinstance(command, AppendText))
            submitted += int(isinstance(command, SubmitRequest))
            cancelled += int(isinstance(command, CancelRequest))
            self._execute_command(command)

        execute(first)
        for _ in range(self.config.max_commands_per_turn - 1):
            try:
                command = self._next_queued_command_nowait()
            except queue.Empty:
                break
            execute(command)
        trace.record_packed_fields(
            "engine_command_drain",
            ("duration_ns", "commands", "appends", "submits", "cancels"),
            (
                time.perf_counter_ns() - trace_started_ns,
                executed,
                appended,
                submitted,
                cancelled,
            ),
        )

    def _run(self, readiness_request: SynthesisRequest) -> None:
        with torch.inference_mode(), self._execution_context():
            self._run_in_context(readiness_request)

    def _run_in_context(self, readiness_request: SynthesisRequest) -> None:
        self._engine_thread_id = threading.get_ident()
        try:
            self._warmup(readiness_request)
            with self._condition:
                self._phase = ServicePhase.READY
                self._condition.notify_all()
            while True:
                command = self._next_command()
                step_result = None
                if command is not None:
                    self._drain_commands(command)
                self._drain_live_input_failures()
                if self._records:
                    step_result = self._step_execution(now_s=time.monotonic())
                    self._poll_records()
                    trace = self._diagnostic_trace()
                    trace_started_ns = (
                        time.perf_counter_ns() if trace is not None and trace.enabled else 0
                    )
                    self._last_health = self._execution_health()
                    if trace is not None and trace.enabled:
                        trace.record_packed_fields(
                            "engine_health_refresh",
                            ("duration_ns",),
                            (time.perf_counter_ns() - trace_started_ns,),
                        )
                if self._stop_requested and not self._records:
                    break
                if command is None and step_result is None and self._records:
                    time.sleep(self.config.idle_poll_interval_s)
        except BaseException as error:
            with self._condition:
                self._failure = error
                self._failed_after_ready = self._phase is ServicePhase.READY
            self._set_metrics(failed=1)
            for record in tuple(self._records.values()):
                record.stream.fail(ServiceUnavailable("Engine thread failed"))
                try:
                    if record.admitted:
                        self._cancel_request(record.request_id)
                        if self._request_is_removable(record.request_id):
                            self._retire(record, completed=False)
                except Exception:
                    pass
            with self._condition:
                self._phase = ServicePhase.FAILED
                self._condition.notify_all()
            return
        with self._condition:
            self._phase = ServicePhase.STOPPED
            self._condition.notify_all()


__all__ = ["Engine", "EngineMetrics"]

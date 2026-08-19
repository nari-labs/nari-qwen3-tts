from __future__ import annotations

import time
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

from nari_qwen3_tts.config import EngineConfig, LiveInputConfig
from nari_qwen3_tts.contract.errors import (
    BackpressureExceeded,
    RequestCancelled,
    ServiceUnavailable,
)
from nari_qwen3_tts.contract.health import EngineHealth, StageStats
from nari_qwen3_tts.contract.request import FragmentTokenization, SynthesisRequest
from nari_qwen3_tts.engine.engine import Engine


@dataclass
class FakeRequestState:
    chunks: tuple[bytes, ...]
    cursor: int = 0
    cancelled: bool = False
    input_finished: bool = True
    routed: int = 0


@dataclass(frozen=True, slots=True)
class FakeRequestSnapshot:
    pcm_chunks: tuple[bytes, ...]
    terminal: bool
    removable: bool
    cancelled: bool


def pcm16(text: str) -> bytes:
    value = text.encode()
    return value if len(value) % 2 == 0 else value + b"\0"


class FakeEngineBackend:
    """Test-only deterministic backend for the concrete Engine shell."""

    def __init__(self) -> None:
        self.requests: dict[str, FakeRequestState] = {}
        self.removed: list[str] = []
        self.routed: list[tuple[str, int, float]] = []
        self.stage_counts = dict.fromkeys(
            ("talker_prefill", "talker_decode", "code_predictor", "codec"), 0
        )
        self.streaming_tokenizer_concurrency = 2

    @staticmethod
    def execution_context():
        return nullcontext()

    def health(self) -> EngineHealth:
        return EngineHealth(
            required_keys=4,
            captured_keys=4,
            required_cuda_graph_instances=4,
            captured_cuda_graph_instances=4,
            capture_failures=0,
            eager_fallbacks=0,
            stages=tuple(
                StageStats(name, count, count, 0)
                for name, count in self.stage_counts.items()
            ),
        )

    def admit(
        self,
        request_id,
        request,
        *,
        admitted_at_s,
        live=False,
        input_finished=True,
    ):
        del admitted_at_s, live
        chunk = pcm16(request.text)
        self.requests[request_id] = FakeRequestState(
            (chunk, chunk),
            input_finished=input_finished,
        )

    def update_request_input(self, request_id, token_ids, *, sequence, is_final):
        del token_ids, sequence
        if is_final:
            self.requests[request_id].input_finished = True

    def tokenize_fragment(self, text, *, is_initial, is_final):
        consumed = len(text) if is_final else max(0, len(text) - 1)
        token_ids = tuple(text[:consumed].encode())
        wrapped_ids = (101, 102, 103, *token_ids, 104) if is_initial and token_ids else ()
        return FragmentTokenization(token_ids, wrapped_ids, consumed)

    def tokenize_streaming_fragment(self, text, *, is_initial, is_final):
        return self.tokenize_fragment(text, is_initial=is_initial, is_final=is_final)

    def prepare_streaming_tokenizer_pool(self) -> None:
        pass

    def step(self, *, now_s):
        del now_s
        for request in self.requests.values():
            if request.cancelled or not request.input_finished or request.cursor >= len(request.chunks):
                continue
            request.cursor += 1
            for name in self.stage_counts:
                self.stage_counts[name] += 1
            return object()
        return None

    def snapshot(self, request_id, *, pcm_start_index=0):
        request = self.requests[request_id]
        return FakeRequestSnapshot(
            pcm_chunks=request.chunks[request.routed + pcm_start_index : request.cursor],
            terminal=request.cursor == len(request.chunks),
            removable=request.cancelled or request.cursor == len(request.chunks),
            cancelled=request.cancelled,
        )

    def record_pcm_routed(self, request_id, *, pcm_bytes, routed_at_s):
        request = self.requests[request_id]
        consumed_chunks = 0
        consumed_bytes = 0
        for chunk in request.chunks[request.routed : request.cursor]:
            consumed_chunks += 1
            consumed_bytes += len(chunk)
            if consumed_bytes >= pcm_bytes:
                break
        if consumed_bytes != pcm_bytes:
            raise RuntimeError("playback credit exceeds committed unrouted PCM or splits a chunk")
        request.routed += consumed_chunks
        self.routed.append((request_id, pcm_bytes, routed_at_s))

    def cancel(self, request_id):
        self.requests[request_id].cancelled = True

    def remove(self, request_id):
        self.removed.append(request_id)
        del self.requests[request_id]


def engine_config(
    *,
    max_active_requests: int = 128,
    max_buffered_bytes: int = 8 * 1024 * 1024,
    command_capacity: int = 512,
    max_pending_text_characters: int = 65_536,
    max_input_append_events: int = 1024,
    max_live_text_tokens: int = 16 * 1024,
    max_update_tokens: int = 4096,
    max_commands_per_turn: int = 32,
    owner_poll_interval_s: float = 0.0005,
) -> EngineConfig:
    return EngineConfig(
        max_active_requests=max_active_requests,
        command_capacity=command_capacity,
        max_commands_per_turn=max_commands_per_turn,
        idle_poll_interval_s=owner_poll_interval_s,
        max_buffered_pcm_bytes=max_buffered_bytes,
        live_input=LiveInputConfig(
            max_pending_text_characters=max_pending_text_characters,
            max_input_append_events=max_input_append_events,
            max_live_text_tokens=max_live_text_tokens,
            max_update_tokens=max_update_tokens,
        ),
    )


class FakeEngine(Engine):
    """Concrete Engine shell with a CPU-only backend, kept entirely in tests."""

    def __init__(self, backend: FakeEngineBackend, *, config: EngineConfig | None = None) -> None:
        self.backend = backend
        self.model = SimpleNamespace(text=backend)
        self._probe_pcm_indexes: dict[str, int] = {}
        self._initialize_loop(config or EngineConfig())

    def _execution_context(self):
        return self.backend.execution_context()

    def _execution_health(self) -> EngineHealth:
        return self.backend.health()

    def _admit_request(
        self,
        request_id,
        request,
        *,
        admitted_at_s,
        live,
        initial_token_ids=(),
        initial_wrapped_ids=(),
        input_finished=True,
    ):
        del initial_token_ids, initial_wrapped_ids
        self.backend.admit(
            request_id,
            request,
            admitted_at_s=admitted_at_s,
            live=live,
        )
        state = self.backend.requests.get(request_id)
        if state is not None:
            state.input_finished = input_finished

    def _update_request_input_batch(self, request_id, updates):
        update_batch = getattr(self.backend, "update_request_input_batch", None)
        if callable(update_batch):
            receipt = update_batch(request_id, updates)
        else:
            receipt = None
            for token_ids, sequence, final in updates:
                value = self.backend.update_request_input(
                    request_id,
                    token_ids,
                    sequence=sequence,
                    is_final=final,
                )
                if value is not None:
                    if receipt is not None:
                        raise RuntimeError("fake backend must expose one atomic update receipt")
                    receipt = value
        if receipt is None:
            receipt = Future()
            receipt.set_result(None)
        return receipt

    def _step_execution(self, *, now_s):
        return self.backend.step(now_s=now_s)

    def _cancel_request(self, request_id):
        self.backend.cancel(request_id)

    def _request_is_removable(self, request_id):
        return self.backend.snapshot(request_id).removable

    def _remove_request(self, request_id):
        self._probe_pcm_indexes.pop(request_id, None)
        self.backend.remove(request_id)

    def _poll_records(self) -> None:
        self._assert_engine_thread()
        for record in tuple(self._records.values()):
            if not record.admitted:
                continue
            snapshot = self.backend.snapshot(
                record.request_id,
                pcm_start_index=self._probe_pcm_indexes.get(record.request_id, 0),
            )
            new_chunks = () if record.cancel_sent else snapshot.pcm_chunks
            for chunk in new_chunks:
                if record.internal_probe:
                    record.ordinary_pcm_bytes += len(chunk)
                    self._probe_pcm_indexes[record.request_id] = (
                        self._probe_pcm_indexes.get(record.request_id, 0) + 1
                    )
                    continue
                try:
                    record.stream.publish(chunk)
                    if chunk:
                        self.backend.record_pcm_routed(
                            record.request_id,
                            pcm_bytes=len(chunk),
                            routed_at_s=time.monotonic(),
                        )
                except BackpressureExceeded:
                    self._set_metrics(backpressured=1)
                    self._request_cancel(
                        record,
                        BackpressureExceeded("request PCM backpressure budget exceeded"),
                    )
                    break
                except Exception as error:
                    self._set_metrics(failed=1)
                    self._request_cancel(
                        record,
                        ServiceUnavailable(f"request PCM delivery failed: {error}"),
                    )
                    break
            if snapshot.cancelled and not record.stream.terminal:
                record.stream.fail(RequestCancelled("request was cancelled"))
            if snapshot.terminal and not record.internal_probe and not record.stream.terminal:
                record.stream.close()
            if snapshot.removable:
                if record.internal_probe:
                    self._ordinary_pcm_bytes = record.ordinary_pcm_bytes
                    self._ordinary_retired = True
                self._retire(record, completed=snapshot.terminal and not snapshot.cancelled)


def request(text: str = "ok") -> SynthesisRequest:
    return SynthesisRequest(text=text, do_sample=False, max_new_tokens=2)

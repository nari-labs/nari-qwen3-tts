from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from nari_qwen3_tts.api.app import create_app
from nari_qwen3_tts.api.http import error_response
from nari_qwen3_tts.api.schemas import SpeechRequestBody, WebSocketStart
from nari_qwen3_tts.api.wav import SAMPLE_RATE, wav_header
from nari_qwen3_tts.api.websocket import WS_PROTOCOL, receive_payload
from nari_qwen3_tts.config import DEFAULT_MODEL_ID
from nari_qwen3_tts.contract.errors import (
    BackpressureExceeded,
    LiveInputClosedError,
    RequestCancelled,
    RequestRejected,
    ServiceCapacityExceeded,
    ServiceUnavailable,
)
from nari_qwen3_tts.contract.stream import PCMStream


def _websocket_endpoint(service):
    app = create_app(service, text_frontend=service.model.text)
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/audio/speech/ws")


class _ScriptedWebSocket:
    def __init__(self, *, offered: bool = True, block_binary: bool = False) -> None:
        self.headers = {"sec-websocket-protocol": WS_PROTOCOL if offered else ""}
        self.inputs: asyncio.Queue[object] = asyncio.Queue()
        self.sent: list[tuple[str, object]] = []
        self.accepted = False
        self.accepted_subprotocol: str | None = None
        self.closed: tuple[int, str] | None = None
        self.block_binary = block_binary
        self.binary_entered = asyncio.Event()
        self.release_binary = asyncio.Event()
        self.concurrent_write = asyncio.Event()
        self._active_writers = 0
        self.max_active_writers = 0

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted = True
        self.accepted_subprotocol = subprotocol

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    async def receive_text(self) -> str:
        item = await self.inputs.get()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return item
        return json.dumps(item)

    async def _begin_write(self) -> None:
        self._active_writers += 1
        self.max_active_writers = max(self.max_active_writers, self._active_writers)
        if self._active_writers > 1:
            self.concurrent_write.set()
        await asyncio.sleep(0)

    def _end_write(self) -> None:
        self._active_writers -= 1

    async def send_json(self, value) -> None:
        await self._begin_write()
        try:
            self.sent.append(("json", value))
        finally:
            self._end_write()

    async def send_bytes(self, value: bytes) -> None:
        await self._begin_write()
        try:
            self.binary_entered.set()
            if self.block_binary:
                await self.release_binary.wait()
            self.sent.append(("bytes", value))
        finally:
            self._end_write()


class _LiveProtocolService:
    def __init__(self, *, stream_bytes: int = 64) -> None:
        self.model = SimpleNamespace(text=self)
        self.streaming_tokenizer_concurrency = 2
        self.config = SimpleNamespace(
            max_websocket_json_characters=65_536,
            input_event_timeout_s=0.2,
            websocket_send_timeout_s=5.0,
            max_active_requests=8,
            live_input=SimpleNamespace(
                max_pending_text_characters=65_536,
                max_input_append_events=1024,
                max_live_text_tokens=16 * 1024,
                max_update_tokens=4096,
            ),
        )
        self.stream = PCMStream(max_buffered_bytes=stream_bytes)
        self.cancelled: list[str] = []
        self.request_id: str | None = None

    def tokenize_streaming_fragment(self, text, *, is_initial, is_final):
        consumed = len(text) if is_final else max(0, len(text) - 1)
        token_ids = tuple(text[:consumed].encode())
        wrapped_ids = (101, 102, 103, *token_ids, 104) if is_initial and token_ids else ()
        return SimpleNamespace(
            token_ids=token_ids,
            wrapped_ids=wrapped_ids,
            consumed_character_count=consumed,
        )

    def prepare_streaming_tokenizer_pool(self) -> None:
        pass

    def begin_live(
        self,
        request_id,
        request,
        *,
        initial_token_ids,
        initial_wrapped_ids,
        input_finished,
        timeout_s=10.0,
    ):
        del request, initial_token_ids, initial_wrapped_ids, timeout_s
        self.request_id = request_id
        self.stream.publish(b"\x01\x00")
        if input_finished:
            self.stream.close()
        return self.stream

    def append_text(self, request_id, token_ids, *, sequence, is_final, timeout_s=10.0):
        del request_id, token_ids, sequence, timeout_s
        if is_final:
            self.stream.close()

    def cancel(self, request_id, *, timeout_s=10.0):
        del timeout_s
        self.cancelled.append(request_id)
        self.stream.fail(RequestCancelled("cancelled"))


def test_websocket_start_uses_server_generated_request_identity() -> None:
    """client request.start does not own the engine request ID."""

    start = WebSocketStart.model_validate(
        {
            "type": "request.start",
            "model": DEFAULT_MODEL_ID,
            "voice": "aiden",
        }
    )
    assert start.to_domain().non_streaming_mode is False

    with pytest.raises(ValidationError, match="request_id|extra"):
        WebSocketStart.model_validate(
            {
                "type": "request.start",
                "request_id": "client-owned",
                "model": DEFAULT_MODEL_ID,
            }
        )


def test_websocket_start_accepts_singular_instruction_and_pcm_aliases() -> None:
    """the WebSocket API preserves its static option vocabulary."""

    for response_format in ("pcm", "pcm16", "pcm_s16le"):
        start = WebSocketStart.model_validate(
            {
                "type": "request.start",
                "model": DEFAULT_MODEL_ID,
                "instruction": "Speak clearly.",
                "response_format": response_format,
            }
        )
        assert start.to_domain().instruct == "Speak clearly."

    with pytest.raises(ValidationError, match="instruction|instruct|extra"):
        WebSocketStart.model_validate(
            {
                "type": "request.start",
                "instruction": "one",
                "instruct": "two",
            }
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_new_tokens", 32_769),
        ("stream_chunk_frames", 129),
        ("stream_first_chunk_frames", 129),
        ("stream_steady_chunk_frames", 129),
        ("stream_chunk_schedule", [1] * 33),
    ],
)
def test_websocket_generation_controls_have_protocol_maxima(field: str, value) -> None:
    """positive but operationally unbounded controls fail at ingress."""

    with pytest.raises((ValidationError, ValueError), match="maximum|less than|32|128|output|chunk"):
        WebSocketStart.model_validate({"type": "request.start", field: value}).to_domain()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_new_tokens", 32_769),
        ("stream_chunk_frames", 129),
        ("stream_first_chunk_frames", 129),
        ("stream_steady_chunk_frames", 129),
        ("stream_chunk_schedule", [1] * 33),
    ],
)
def test_http_generation_controls_bound_collected_response_ownership(
    field: str,
    value,
) -> None:
    """collected responses have the same explicit generation bounds."""

    with pytest.raises((ValidationError, ValueError), match="maximum|less than|32|128|output|chunk"):
        SpeechRequestBody.model_validate({"input": "hello", field: value}).to_domain()


def test_missing_websocket_protocol_is_accepted_then_closed_1002() -> None:
    """ASGI must expose the protocol close code instead of HTTP 403."""

    async def scenario() -> _ScriptedWebSocket:
        socket = _ScriptedWebSocket(offered=False)
        await _websocket_endpoint(_LiveProtocolService())(socket)
        return socket

    socket = asyncio.run(scenario())
    assert socket.accepted
    assert socket.accepted_subprotocol is None
    assert socket.closed is not None and socket.closed[0] == 1002


def test_session_created_has_exact_audio_metadata() -> None:
    """the first server event completely describes binary frames."""

    async def scenario() -> _ScriptedWebSocket:
        socket = _ScriptedWebSocket()
        await socket.inputs.put(WebSocketDisconnect())
        await _websocket_endpoint(_LiveProtocolService())(socket)
        return socket

    socket = asyncio.run(scenario())
    assert socket.sent[0] == (
        "json",
        {
            "type": "session.created",
            "protocol": WS_PROTOCOL,
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
            },
        },
    )


def test_websocket_audio_starts_before_input_end() -> None:
    """a stable append enables full-duplex audio immediately."""

    async def scenario() -> tuple[_ScriptedWebSocket, _LiveProtocolService]:
        service = _LiveProtocolService()
        socket = _ScriptedWebSocket()
        endpoint = _websocket_endpoint(service)
        task = asyncio.create_task(endpoint(socket))
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await asyncio.wait_for(socket.binary_entered.wait(), timeout=0.25)
        await socket.inputs.put({"type": "input_text.end", "sequence": 1})
        await asyncio.sleep(0)
        await socket.inputs.put(WebSocketDisconnect())
        await asyncio.wait_for(task, timeout=1)
        return socket, service

    socket, _service = asyncio.run(scenario())
    sent_types = [value.get("type") for kind, value in socket.sent if kind == "json"]
    assert "response.started" in sent_types
    assert any(kind == "bytes" for kind, _value in socket.sent)


def test_websocket_tokenizes_off_engine_and_passes_typed_initial_and_update_tokens() -> None:
    class TokenAwareService(_LiveProtocolService):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(text=self)
            self.streaming_tokenizer_concurrency = 2
            self.pool_prepare_threads: list[int] = []
            self.tokenizer_threads: list[int] = []
            self.begin_values: list[tuple[object, ...]] = []
            self.update_values: list[tuple[object, ...]] = []

        def prepare_streaming_tokenizer_pool(self) -> None:
            self.pool_prepare_threads.append(threading.get_ident())

        def tokenize_streaming_fragment(self, text, *, is_initial, is_final):
            assert self.pool_prepare_threads
            self.tokenizer_threads.append(threading.get_ident())
            if is_initial and not is_final:
                consumed = len(text) - 1
                return SimpleNamespace(
                    token_ids=(11,),
                    wrapped_ids=(101, 102, 103, 11, 104),
                    consumed_character_count=consumed,
                )
            return SimpleNamespace(
                token_ids=(12,),
                wrapped_ids=(),
                consumed_character_count=len(text),
            )

        def begin_live(
            self,
            request_id,
            request,
            *,
            initial_token_ids,
            initial_wrapped_ids,
            input_finished,
            timeout_s=10.0,
        ):
            del timeout_s
            self.request_id = request_id
            self.begin_values.append(
                (
                    request.text,
                    tuple(initial_token_ids),
                    tuple(initial_wrapped_ids),
                    input_finished,
                )
            )
            self.stream.publish(b"\x01\x00")
            return self.stream

        def append_text(self, request_id, token_ids, *, sequence, is_final, timeout_s=10.0):
            del request_id, timeout_s
            self.update_values.append((tuple(token_ids), sequence, is_final))
            if is_final:
                self.stream.close()

    async def scenario():
        service = TokenAwareService()
        event_loop_thread = threading.get_ident()
        socket = _ScriptedWebSocket()
        endpoint = _websocket_endpoint(service)
        task = asyncio.create_task(endpoint(socket))
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await asyncio.wait_for(socket.binary_entered.wait(), timeout=0.25)
        await socket.inputs.put({"type": "input_text.end", "sequence": 1})
        await asyncio.wait_for(task, timeout=1)
        return service, event_loop_thread

    service, event_loop_thread = asyncio.run(scenario())

    assert service.begin_values == [("hello ", (11,), (101, 102, 103, 11, 104), False)]
    assert service.update_values == [((12,), 0, True)]
    assert service.pool_prepare_threads
    assert service.tokenizer_threads
    assert all(
        thread_id != event_loop_thread
        for thread_id in (*service.pool_prepare_threads, *service.tokenizer_threads)
    )


def test_websocket_splits_large_tokenized_updates_before_engine_queueing() -> None:
    class SplitService(_LiveProtocolService):
        def __init__(self) -> None:
            super().__init__()
            self.config.live_input.max_update_tokens = 2
            self.updates: list[tuple[tuple[int, ...], int, bool]] = []

        def tokenize_streaming_fragment(self, text, *, is_initial, is_final):
            if is_initial and not is_final:
                return SimpleNamespace(
                    token_ids=(11,),
                    wrapped_ids=(101, 102, 103, 11, 104),
                    consumed_character_count=len(text) - 1,
                )
            return SimpleNamespace(
                token_ids=(12, 13, 14, 15, 16),
                wrapped_ids=(),
                consumed_character_count=len(text),
            )

        def append_text(self, request_id, token_ids, *, sequence, is_final, timeout_s=10.0):
            self.updates.append((tuple(token_ids), sequence, is_final))
            return super().append_text(
                request_id,
                token_ids,
                sequence=sequence,
                is_final=is_final,
                timeout_s=timeout_s,
            )

    async def scenario() -> SplitService:
        service = SplitService()
        socket = _ScriptedWebSocket()
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await socket.inputs.put({"type": "input_text.end", "sequence": 1})
        await _websocket_endpoint(service)(socket)
        return service

    service = asyncio.run(scenario())
    assert service.updates == [
        ((12, 13), 0, False),
        ((14, 15), 1, False),
        ((16,), 2, True),
    ]


def test_websocket_cumulative_text_limit_includes_consumed_fragments() -> None:
    async def scenario() -> _ScriptedWebSocket:
        service = _LiveProtocolService()
        service.config.live_input.max_pending_text_characters = 3
        socket = _ScriptedWebSocket()
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "ab "})
        await socket.inputs.put({"type": "input_text.append", "sequence": 1, "text": "c"})
        await _websocket_endpoint(service)(socket)
        return socket

    socket = asyncio.run(scenario())
    error = next(value for kind, value in socket.sent if kind == "json" and value["type"] == "error")
    assert error["error"]["code"] == "input_too_large"
    assert socket.closed is not None and socket.closed[0] == 1009


def test_websocket_nonfinal_tokenization_must_retain_mutable_tail() -> None:
    class UnsafeTokenizerService(_LiveProtocolService):
        def tokenize_streaming_fragment(self, text, *, is_initial, is_final):
            del is_final
            token_ids = tuple(text.encode())
            return SimpleNamespace(
                token_ids=token_ids,
                wrapped_ids=(101, 102, 103, *token_ids, 104) if is_initial else (),
                consumed_character_count=len(text),
            )

    async def scenario() -> _ScriptedWebSocket:
        socket = _ScriptedWebSocket()
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "unsafe"})
        await _websocket_endpoint(UnsafeTokenizerService())(socket)
        return socket

    socket = asyncio.run(scenario())
    error = next(value for kind, value in socket.sent if kind == "json" and value["type"] == "error")
    assert error["error"]["code"] == "invalid_request"
    assert "unfinished tail" in error["error"]["message"]


def test_websocket_ack_started_done_and_close_match_protocol() -> None:
    """control frames retain exact causal identity."""

    async def scenario() -> _ScriptedWebSocket:
        service = _LiveProtocolService()
        socket = _ScriptedWebSocket()
        endpoint = _websocket_endpoint(service)
        task = asyncio.create_task(endpoint(socket))
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await asyncio.wait_for(socket.binary_entered.wait(), timeout=0.25)
        await socket.inputs.put({"type": "input_text.end", "sequence": 1})
        await asyncio.wait_for(task, timeout=1)
        return socket

    socket = asyncio.run(scenario())
    controls = [value for kind, value in socket.sent if kind == "json"]
    configured = next(value for value in controls if value["type"] == "request.configured")
    started = next(value for value in controls if value["type"] == "response.started")
    append_ack = next(
        value
        for value in controls
        if value["type"] == "input_text.ack" and value["sequence"] == 0
    )
    end_ack = next(
        value
        for value in controls
        if value["type"] == "input_text.ack" and value["sequence"] == 1
    )
    done = next(value for value in controls if value["type"] == "response.done")

    assert configured == {"type": "request.configured"}
    assert started["request_id"]
    assert started["audio"] == {
        "encoding": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
    }
    assert append_ack == {
        "type": "input_text.ack",
        "event": "input_text.append",
        "sequence": 0,
        "request_id": started["request_id"],
    }
    assert end_ack == {
        "type": "input_text.ack",
        "event": "input_text.end",
        "sequence": 1,
        "request_id": started["request_id"],
    }
    assert done["request_id"] == started["request_id"]
    assert done["stop_reason"]
    assert done["audio_chunks"] == 1
    assert socket.closed is not None and socket.closed[0] == 1000


def test_websocket_append_ack_waits_for_engine_queue_acceptance() -> None:
    class DeferredService(_LiveProtocolService):
        def __init__(self) -> None:
            super().__init__()
            self.append_entered = threading.Event()
            self.allow_append = threading.Event()

        def append_text(self, request_id, token_ids, *, sequence, is_final, timeout_s=10.0):
            if sequence == 0:
                self.append_entered.set()
                assert self.allow_append.wait(timeout=1)
            return super().append_text(
                request_id,
                token_ids,
                sequence=sequence,
                is_final=is_final,
                timeout_s=timeout_s,
            )

    async def scenario() -> _ScriptedWebSocket:
        service = DeferredService()
        socket = _ScriptedWebSocket()
        endpoint = _websocket_endpoint(service)
        task = asyncio.create_task(endpoint(socket))
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await asyncio.wait_for(socket.binary_entered.wait(), timeout=0.25)
        await socket.inputs.put({"type": "input_text.append", "sequence": 1, "text": "world "})
        assert await asyncio.to_thread(service.append_entered.wait, 1)
        await asyncio.sleep(0.05)
        assert not any(
            kind == "json"
            and value.get("type") == "input_text.ack"
            and value.get("sequence") == 1
            for kind, value in socket.sent
        )
        service.allow_append.set()
        for _ in range(100):
            if any(
                kind == "json"
                and value.get("type") == "input_text.ack"
                and value.get("sequence") == 1
                for kind, value in socket.sent
            ):
                break
            await asyncio.sleep(0.001)
        await socket.inputs.put(WebSocketDisconnect())
        await asyncio.wait_for(task, timeout=1)
        return socket

    socket = asyncio.run(scenario())
    assert any(
        kind == "json"
        and value.get("type") == "input_text.ack"
        and value.get("sequence") == 1
        for kind, value in socket.sent
    )


def test_terminal_live_input_race_is_not_reported_as_invalid_request() -> None:
    class TerminalService(_LiveProtocolService):
        def begin_live(
            self,
            request_id,
            request,
            *,
            initial_token_ids,
            initial_wrapped_ids,
            input_finished,
            timeout_s=10.0,
        ):
            del (
                request_id,
                request,
                initial_token_ids,
                initial_wrapped_ids,
                input_finished,
                timeout_s,
            )
            raise LiveInputClosedError("generation finished before live input publication")

    async def scenario() -> _ScriptedWebSocket:
        socket = _ScriptedWebSocket()
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await _websocket_endpoint(TerminalService())(socket)
        return socket

    socket = asyncio.run(scenario())
    error = next(value for kind, value in socket.sent if kind == "json" and value["type"] == "error")
    assert error["error"] == {
        "type": "generation_error",
        "code": "live_input_closed",
        "message": "generation finished before live input publication",
    }
    assert socket.closed is not None and socket.closed[0] == 1000


def test_websocket_protocol_errors_are_structured_terminal_frames() -> None:
    """invalid client input has stable code/message/close semantics."""

    async def scenario() -> _ScriptedWebSocket:
        socket = _ScriptedWebSocket()
        await socket.inputs.put({"type": "unknown.event"})
        await _websocket_endpoint(_LiveProtocolService())(socket)
        return socket

    socket = asyncio.run(scenario())
    error = next(value for kind, value in socket.sent if kind == "json" and value["type"] == "error")
    assert error["request_id"] is None
    assert error["error"]["type"] == "invalid_request_error"
    assert error["error"]["code"] == "unknown_event"
    assert error["error"]["message"]
    assert socket.closed is not None and socket.closed[0] == 1008


def test_websocket_has_one_socket_writer_during_cancel_and_audio() -> None:
    """receiver and output paths never write concurrently."""

    async def scenario() -> _ScriptedWebSocket:
        service = _LiveProtocolService()
        socket = _ScriptedWebSocket(block_binary=True)
        endpoint = _websocket_endpoint(service)
        task = asyncio.create_task(endpoint(socket))
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await asyncio.wait_for(socket.binary_entered.wait(), timeout=0.25)
        await socket.inputs.put({"type": "response.cancel"})
        try:
            await asyncio.wait_for(socket.concurrent_write.wait(), timeout=0.05)
        except TimeoutError:
            pass
        socket.release_binary.set()
        await socket.inputs.put(WebSocketDisconnect())
        await asyncio.wait_for(task, timeout=1)
        return socket

    socket = asyncio.run(scenario())
    assert socket.max_active_writers == 1


def test_blocked_websocket_send_remains_in_outstanding_byte_budget() -> None:
    """dequeuing a frame does not free its budget before ASGI send."""

    async def scenario() -> None:
        service = _LiveProtocolService(stream_bytes=2)
        socket = _ScriptedWebSocket(block_binary=True)
        endpoint = _websocket_endpoint(service)
        task = asyncio.create_task(endpoint(socket))
        await socket.inputs.put({"type": "request.start", "model": DEFAULT_MODEL_ID})
        await socket.inputs.put({"type": "input_text.append", "sequence": 0, "text": "hello "})
        await asyncio.wait_for(socket.binary_entered.wait(), timeout=0.25)
        with pytest.raises(BackpressureExceeded):
            service.stream.publish(b"\x02\x00")
        socket.release_binary.set()
        await socket.inputs.put(WebSocketDisconnect())
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(scenario())


def test_websocket_message_limit_counts_utf8_bytes() -> None:
    """a multibyte client frame cannot bypass the byte limit."""

    class Socket:
        @staticmethod
        async def receive_text() -> str:
            return json.dumps({"type": "x", "text": "\u00e9" * 16}, ensure_ascii=False)

    with pytest.raises(ValueError, match="byte|capacity|large"):
        asyncio.run(receive_payload(Socket(), max_characters=32))


@pytest.mark.parametrize("pcm_bytes", [1, 3, 2**32, 2**32 - 1])
def test_wav_header_rejects_unaligned_or_unrepresentable_pcm(pcm_bytes: int) -> None:
    """RIFF lengths and PCM16 sample alignment fail closed."""

    with pytest.raises(ValueError, match="PCM|even|RIFF|represent"):
        wav_header(pcm_bytes=pcm_bytes)


@pytest.mark.parametrize(
    "error,status",
    [
        (ServiceUnavailable("x"), 503),
        (ServiceCapacityExceeded("x"), 429),
        (RequestRejected("x"), 422),
        (ValueError("secret"), 500),
        (TypeError("secret"), 500),
        (RequestCancelled("x"), 499),
        (BackpressureExceeded("x"), 503),
        (RuntimeError("secret"), 500),
    ],
)
def test_http_error_mapping_is_complete_and_internal_safe(error, status: int) -> None:
    """all typed failures have deterministic public status."""

    response = error_response(error)
    assert response.status_code == status
    if status == 500:
        assert "secret" not in str(response.detail)


def test_http_instruction_aliases_have_one_domain_value() -> None:
    """equal aliases agree; conflicting aliases fail before admission."""

    body = SpeechRequestBody(input="hello", instruct="clear", instructions="clear")
    assert body.to_domain().instruct == "clear"

    with pytest.raises(ValueError, match="disagree"):
        SpeechRequestBody(input="hello", instruct="one", instructions="two").to_domain()

"""WebSocket wire protocol and single-writer helpers."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, replace
from typing import Literal

from fastapi import WebSocket, WebSocketDisconnect

from nari_qwen3_tts.api.schemas import WebSocketAppend, WebSocketCancel, WebSocketEnd, WebSocketStart
from nari_qwen3_tts.api.wav import SAMPLE_RATE
from nari_qwen3_tts.config import ApiConfig
from nari_qwen3_tts.contract.errors import (
    BackpressureExceeded,
    LiveInputClosedError,
    RequestCancelled,
    StreamingTextControlTokenError,
)
from nari_qwen3_tts.contract.request import FragmentTokenization, SynthesisRequest, TextFrontend
from nari_qwen3_tts.contract.stream import PCMStream
from nari_qwen3_tts.engine.engine import Engine

WS_PROTOCOL = "nari.speech.v1"
_NORMAL_FRAME_CAPACITY = 32


class WebSocketProtocolError(ValueError):
    def __init__(self, code: str, message: str, *, close_code: int = 1008) -> None:
        super().__init__(message)
        self.code = code
        self.close_code = close_code


def structured_error(
    request_id: str | None,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> dict[str, object]:
    return {
        "type": "error",
        "request_id": request_id,
        "error": {
            "type": error_type,
            "code": code,
            "message": message or code,
        },
    }


@dataclass(frozen=True, slots=True)
class OutboundFrame:
    kind: Literal["json", "bytes", "close"]
    value: object
    normal: bool = True
    stream: PCMStream | None = None


async def send_outbound(
    websocket: WebSocket,
    frames: asyncio.Queue[OutboundFrame],
    normal_slots: asyncio.Semaphore,
) -> None:
    """Be the connection's only ASGI socket writer."""

    while True:
        frame = await frames.get()
        try:
            if frame.kind == "json":
                await websocket.send_json(frame.value)
            elif frame.kind == "bytes":
                await websocket.send_bytes(frame.value)
            else:
                code, reason = frame.value
                await websocket.close(code=code, reason=reason)
                return
        finally:
            if frame.stream is not None:
                frame.stream.acknowledge(frame.value)
            if frame.normal:
                normal_slots.release()
            frames.task_done()


async def queue_normal(
    frames: asyncio.Queue[OutboundFrame],
    normal_slots: asyncio.Semaphore,
    frame: OutboundFrame,
    *,
    timeout_s: float,
) -> None:
    await asyncio.wait_for(normal_slots.acquire(), timeout=timeout_s)
    try:
        await frames.put(frame)
    except BaseException:
        normal_slots.release()
        raise


async def stream_output(
    stream: PCMStream,
    frames: asyncio.Queue[OutboundFrame],
    normal_slots: asyncio.Semaphore,
    *,
    timeout_s: float,
) -> int:
    chunks = 0
    while True:
        value = await asyncio.to_thread(stream.acquire, timeout_s=None)
        if value is None:
            return chunks
        try:
            await queue_normal(
                frames,
                normal_slots,
                OutboundFrame("bytes", value, stream=stream),
                timeout_s=timeout_s,
            )
        except BaseException:
            stream.acknowledge(value)
            raise
        chunks += 1


def offered_protocols(websocket: WebSocket) -> tuple[str, ...]:
    header = websocket.headers.get("sec-websocket-protocol", "")
    return tuple(value.strip() for value in header.split(",") if value.strip())


async def receive_payload(websocket: WebSocket, *, max_characters: int) -> dict[str, object]:
    raw = await websocket.receive_text()
    if len(raw.encode("utf-8")) > max_characters:
        raise ValueError("WebSocket JSON message byte capacity is too large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("WebSocket event must be a JSON object")
    return value


class _LiveInputSession:
    """Own one connection's protocol state without leaking it into the Engine."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        engine: Engine,
        api_config: ApiConfig,
        model_id: str,
        text_frontend: TextFrontend,
        tokenizer_slots: asyncio.Semaphore,
    ) -> None:
        self.websocket = websocket
        self.engine = engine
        self.api_config = api_config
        self.model_id = model_id
        self.text_frontend = text_frontend
        self.tokenizer_slots = tokenizer_slots
        self.frames: asyncio.Queue[OutboundFrame] = asyncio.Queue(
            maxsize=_NORMAL_FRAME_CAPACITY + 2
        )
        self.normal_slots = asyncio.Semaphore(_NORMAL_FRAME_CAPACITY)
        self.send_timeout_s = api_config.websocket_send_timeout_s
        self.sender = asyncio.create_task(
            send_outbound(websocket, self.frames, self.normal_slots)
        )
        self.request_id: str | None = None
        self.request: SynthesisRequest | None = None
        self.stream: PCMStream | None = None
        self.input_finished = False
        self.submitted = False
        self.response_started = False
        self.output_task: asyncio.Task[int] | None = None
        self.terminal_enqueued = False
        self.expected_client_sequence = 0
        self.update_sequence = 0
        self.append_events = 0
        self.total_characters = 0
        self.all_text = ""
        self.pending_text = ""
        self.committed_token_count = 0
        self.live_config = engine.config.live_input

    async def _queue_json(self, value: dict[str, object]) -> None:
        await queue_normal(
            self.frames,
            self.normal_slots,
            OutboundFrame("json", value),
            timeout_s=self.send_timeout_s,
        )

    async def _finish(self, value: dict[str, object], *, close_code: int) -> None:
        await self.frames.put(OutboundFrame("json", value, normal=False))
        await self.frames.put(
            OutboundFrame(
                "close",
                (close_code, str(value.get("type", "terminal"))),
                normal=False,
            )
        )

    def _consume_sequence(self, sequence: int) -> None:
        if sequence != self.expected_client_sequence:
            raise WebSocketProtocolError(
                "invalid_sequence",
                f"expected sequence {self.expected_client_sequence}, got {sequence}",
            )
        self.expected_client_sequence += 1

    @staticmethod
    def _validated_ids(values: object, *, name: str) -> tuple[int, ...]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, (tuple, list)) or any(
            type(value) is not int or value < 0 for value in values
        ):
            raise RuntimeError(f"streaming tokenizer returned invalid {name}")
        return tuple(values)

    async def _tokenize_pending(self, *, is_final: bool) -> FragmentTokenization:
        try:
            async with self.tokenizer_slots:
                value = await asyncio.to_thread(
                    self.text_frontend.tokenize_streaming_fragment,
                    self.pending_text,
                    is_initial=not self.submitted,
                    is_final=is_final,
                )
        except StreamingTextControlTokenError as error:
            raise WebSocketProtocolError("invalid_text", str(error)) from error
        token_ids = self._validated_ids(getattr(value, "token_ids", None), name="token IDs")
        wrapped_ids = self._validated_ids(
            getattr(value, "wrapped_ids", None),
            name="wrapped IDs",
        )
        consumed = getattr(value, "consumed_character_count", None)
        if type(consumed) is not int or not 0 <= consumed <= len(self.pending_text):
            raise RuntimeError("streaming tokenizer returned an invalid consumed character count")
        if is_final and consumed != len(self.pending_text):
            raise RuntimeError("final streaming tokenization did not flush its mutable tail")
        if not is_final and consumed == len(self.pending_text):
            raise RuntimeError("non-final streaming tokenization must retain an unfinished tail")
        if consumed == 0 and token_ids:
            raise RuntimeError("streaming tokenizer returned IDs without consuming text")
        if self.submitted and wrapped_ids:
            raise RuntimeError("non-initial streaming fragments cannot include wrapped IDs")
        if not self.submitted and token_ids and len(wrapped_ids) < 4:
            raise RuntimeError("initial streaming fragments require wrapped IDs")
        if self.committed_token_count + len(token_ids) > self.live_config.max_live_text_tokens:
            raise WebSocketProtocolError(
                "input_too_large",
                "live text token capacity exceeded",
                close_code=1009,
            )
        return FragmentTokenization(token_ids, wrapped_ids, consumed)

    async def _start_output(self) -> None:
        if self.response_started or self.request_id is None or self.stream is None:
            return
        self.response_started = True
        await self._queue_json(
            {
                "type": "response.started",
                "request_id": self.request_id,
                "audio": {
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                },
            }
        )
        self.output_task = asyncio.create_task(
            stream_output(
                self.stream,
                self.frames,
                self.normal_slots,
                timeout_s=self.send_timeout_s,
            )
        )

    async def _submit_initial(
        self,
        fragment: FragmentTokenization,
        *,
        finished: bool,
    ) -> None:
        if self.request_id is None or self.request is None:
            raise RuntimeError("cannot admit live input before request.start")
        self.stream = await asyncio.to_thread(
            self.engine.begin_live,
            self.request_id,
            replace(self.request, text=self.all_text),
            initial_token_ids=fragment.token_ids,
            initial_wrapped_ids=fragment.wrapped_ids,
            input_finished=finished,
            timeout_s=self.api_config.command_timeout_s,
        )
        self.submitted = True
        await self._start_output()

    async def _submit_update(self, token_ids: tuple[int, ...], *, finished: bool) -> None:
        if self.request_id is None or not self.submitted:
            raise RuntimeError("cannot update live input before initial admission")
        chunks = [
            token_ids[offset : offset + self.live_config.max_update_tokens]
            for offset in range(0, len(token_ids), self.live_config.max_update_tokens)
        ] or [()]
        for index, chunk in enumerate(chunks):
            await asyncio.to_thread(
                self.engine.append_text,
                self.request_id,
                tuple(chunk),
                sequence=self.update_sequence,
                is_final=finished and index == len(chunks) - 1,
                timeout_s=self.api_config.command_timeout_s,
            )
            self.update_sequence += 1

    async def _finish_output(self, stop_reason: str) -> None:
        if self.terminal_enqueued:
            return
        self.terminal_enqueued = True
        chunks = 0
        if self.output_task is not None:
            try:
                chunks = await self.output_task
            except RequestCancelled:
                stop_reason = "cancelled"
            except BackpressureExceeded as error:
                await self._finish(
                    structured_error(self.request_id, "backpressure", str(error)),
                    close_code=1011,
                )
                return
            except Exception as error:
                await self._finish(
                    structured_error(self.request_id, "generation_failed", str(error)),
                    close_code=1011,
                )
                return
        await self._finish(
            {
                "type": "response.done",
                "request_id": self.request_id,
                "stop_reason": stop_reason,
                "audio_chunks": chunks,
            },
            close_code=1000,
        )

    async def _configure(self, payload: dict[str, object]) -> None:
        if self.request_id is not None:
            raise ValueError("WebSocket request was already configured")
        start = WebSocketStart.model_validate(payload)
        self.request_id = f"ws-{uuid.uuid4().hex}"
        self.request = start.to_domain(served_model_id=self.model_id)
        await asyncio.to_thread(self.text_frontend.prepare_streaming_tokenizer_pool)
        await self._queue_json({"type": "request.configured"})

    async def _append(self, payload: dict[str, object]) -> None:
        if self.request_id is None or self.input_finished:
            raise ValueError("WebSocket text append is not currently accepted")
        append = WebSocketAppend.model_validate(payload)
        self._consume_sequence(append.sequence)
        self.total_characters += len(append.text)
        self.append_events += 1
        if self.total_characters > self.live_config.max_pending_text_characters:
            raise WebSocketProtocolError(
                "input_too_large",
                "live text character capacity exceeded",
                close_code=1009,
            )
        if self.append_events > self.live_config.max_input_append_events:
            raise WebSocketProtocolError(
                "too_many_input_events",
                "live text append-event capacity exceeded",
            )
        self.all_text += append.text
        self.pending_text += append.text
        if len(self.pending_text) > self.live_config.max_pending_text_characters:
            raise WebSocketProtocolError(
                "input_too_large",
                "pending live text capacity exceeded",
                close_code=1009,
            )
        fragment = await self._tokenize_pending(is_final=False)
        if fragment.consumed_character_count:
            self.pending_text = self.pending_text[fragment.consumed_character_count :]
            if self.submitted:
                await self._submit_update(fragment.token_ids, finished=False)
            else:
                await self._submit_initial(fragment, finished=False)
            self.committed_token_count += len(fragment.token_ids)
        await self._queue_json(
            {
                "type": "input_text.ack",
                "event": "input_text.append",
                "sequence": append.sequence,
                "request_id": self.request_id if self.submitted else None,
            }
        )

    async def _end_input(self, payload: dict[str, object]) -> None:
        if self.request_id is None or self.input_finished:
            raise ValueError("WebSocket text input cannot end in the current state")
        end = WebSocketEnd.model_validate(payload)
        self._consume_sequence(end.sequence)
        self.input_finished = True
        fragment = await self._tokenize_pending(is_final=True)
        self.pending_text = ""
        if not self.committed_token_count and not fragment.token_ids:
            raise WebSocketProtocolError("empty_input", "live text input produced no tokens")
        if self.submitted:
            await self._submit_update(fragment.token_ids, finished=True)
        else:
            await self._submit_initial(fragment, finished=True)
        self.committed_token_count += len(fragment.token_ids)
        await self._queue_json(
            {
                "type": "input_text.ack",
                "event": "input_text.end",
                "sequence": end.sequence,
                "request_id": self.request_id,
            }
        )
        await self._finish_output("stop")
        await self.sender

    async def _cancel(self, payload: dict[str, object]) -> None:
        if self.request_id is None:
            raise ValueError("WebSocket request is not configured")
        WebSocketCancel.model_validate(payload)
        if self.submitted:
            await asyncio.to_thread(
                self.engine.cancel,
                self.request_id,
                timeout_s=self.api_config.command_timeout_s,
            )
        await self._finish_output("cancelled")
        await self.sender

    async def _receive_events(self) -> None:
        while True:
            payload = await asyncio.wait_for(
                receive_payload(
                    self.websocket,
                    max_characters=self.api_config.max_websocket_json_characters,
                ),
                timeout=self.api_config.input_event_timeout_s,
            )
            event_type = payload.get("type")
            if event_type == "request.start":
                await self._configure(payload)
            elif event_type == "input_text.append":
                await self._append(payload)
            elif event_type == "input_text.end":
                await self._end_input(payload)
                return
            elif event_type == "response.cancel":
                await self._cancel(payload)
                return
            else:
                raise WebSocketProtocolError(
                    "unknown_event",
                    f"unsupported WebSocket event type: {event_type!r}",
                )

    async def _cleanup(self) -> None:
        if (
            self.request_id is not None
            and self.submitted
            and self.stream is not None
            and not self.stream.terminal
        ):
            try:
                await asyncio.to_thread(
                    self.engine.cancel,
                    self.request_id,
                    timeout_s=self.api_config.command_timeout_s,
                )
            except Exception:
                pass
        if self.output_task is not None:
            if not self.output_task.done():
                self.output_task.cancel()
            await asyncio.gather(self.output_task, return_exceptions=True)
        if not self.sender.done():
            if self.terminal_enqueued:
                try:
                    await asyncio.wait_for(self.sender, timeout=self.send_timeout_s)
                except TimeoutError:
                    self.sender.cancel()
            else:
                self.sender.cancel()
        await asyncio.gather(self.sender, return_exceptions=True)

    async def run(self) -> None:
        await self._queue_json(
            {
                "type": "session.created",
                "protocol": WS_PROTOCOL,
                "audio": {
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                },
            }
        )
        await self.frames.join()
        try:
            await self._receive_events()
        except WebSocketDisconnect:
            pass
        except TimeoutError as error:
            if not self.terminal_enqueued:
                self.terminal_enqueued = True
                await self._finish(
                    structured_error(self.request_id, "input_timeout", str(error)),
                    close_code=1008,
                )
        except WebSocketProtocolError as error:
            if not self.terminal_enqueued:
                self.terminal_enqueued = True
                await self._finish(
                    structured_error(self.request_id, error.code, str(error)),
                    close_code=error.close_code,
                )
        except LiveInputClosedError as error:
            if not self.terminal_enqueued:
                self.terminal_enqueued = True
                await self._finish(
                    structured_error(
                        self.request_id,
                        "live_input_closed",
                        str(error),
                        error_type="generation_error",
                    ),
                    close_code=1000,
                )
        except BaseException as error:
            if not self.terminal_enqueued:
                self.terminal_enqueued = True
                close_code = 1009 if "capacity" in str(error).lower() else 1008
                await self._finish(
                    structured_error(self.request_id, "invalid_request", str(error)),
                    close_code=close_code,
                )
        finally:
            await self._cleanup()


async def handle_speech_websocket(
    websocket: WebSocket,
    *,
    engine: Engine,
    api_config: ApiConfig,
    model_id: str,
    tokenizer: TextFrontend,
    tokenizer_slots: asyncio.Semaphore,
) -> None:
    if WS_PROTOCOL not in offered_protocols(websocket):
        await websocket.accept()
        await websocket.close(code=1002, reason=f"required subprotocol: {WS_PROTOCOL}")
        return
    await websocket.accept(subprotocol=WS_PROTOCOL)
    await _LiveInputSession(
        websocket,
        engine=engine,
        api_config=api_config,
        model_id=model_id,
        text_frontend=tokenizer,
        tokenizer_slots=tokenizer_slots,
    ).run()


__all__ = [
    "OutboundFrame",
    "WS_PROTOCOL",
    "WebSocketProtocolError",
    "handle_speech_websocket",
    "offered_protocols",
    "queue_normal",
    "receive_payload",
    "send_outbound",
    "stream_output",
    "structured_error",
]

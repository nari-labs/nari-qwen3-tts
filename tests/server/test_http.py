from __future__ import annotations

import json
import struct

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nari_qwen3_tts.api.app import create_app
from nari_qwen3_tts.api.websocket import WS_PROTOCOL
from nari_qwen3_tts.config import DEFAULT_MODEL_ID
from nari_qwen3_tts.contract.errors import RequestRejected
from nari_qwen3_tts.contract.health import ServicePhase
from server.fakes import FakeEngine, FakeEngineBackend, engine_config, request


def _client():
    service = FakeEngine(FakeEngineBackend(), config=engine_config(max_buffered_bytes=128))
    service.start(request("p"), timeout_s=2)
    return service, TestClient(create_app(service, text_frontend=service.model.text))


def test_liveness_is_not_readiness_before_ordinary_tts_probe() -> None:
    service = FakeEngine(FakeEngineBackend(), config=engine_config(max_buffered_bytes=128))
    client = TestClient(create_app(service, text_frontend=service.model.text))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503


def test_health_readiness_models_and_strict_http_pcm() -> None:
    service, client = _client()
    try:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True
        assert client.get("/v1/models").json()["data"][0]["id"] == DEFAULT_MODEL_ID

        response = client.post(
            "/v1/audio/speech",
            json={
                "model": DEFAULT_MODEL_ID,
                "input": "ab",
                "voice": "aiden",
                "language": "english",
                "response_format": "pcm",
                "do_sample": False,
                "max_new_tokens": 2,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/pcm")
        assert response.headers["x-nari-request-id"].startswith("http-")
        assert response.content == b"abab"
        assert client.post(
            "/v1/audio/speech",
            json={"model": "wrong", "input": "x", "voice": "aiden"},
        ).status_code == 422
        assert client.post(
            "/v1/audio/speech",
            json={"model": DEFAULT_MODEL_ID, "input": "x", "voice": "aiden", "extra": 1},
        ).status_code == 422
    finally:
        service.stop(timeout_s=2)


def test_http_maps_uncaptured_request_shape_to_422_without_losing_readiness() -> None:
    class RejectingPort(FakeEngineBackend):
        def admit(self, request_id, request, *, admitted_at_s, live=False):
            if request.text == "unsupported":
                raise RequestRejected("Codec capture does not cover request")
            return super().admit(
                request_id,
                request,
                admitted_at_s=admitted_at_s,
                live=live,
            )

    service = FakeEngine(
        RejectingPort(),
        config=engine_config(max_buffered_bytes=128),
    )
    service.start(request("probe"), timeout_s=2)
    client = TestClient(create_app(service, text_frontend=service.model.text))
    try:
        rejected = client.post(
            "/v1/audio/speech",
            json={
                "model": DEFAULT_MODEL_ID,
                "input": "unsupported",
                "voice": "aiden",
                "response_format": "pcm",
                "do_sample": False,
                "max_new_tokens": 2,
            },
        )
        assert rejected.status_code == 422
        assert "Codec capture" in rejected.json()["detail"]
        assert client.get("/ready").status_code == 200

        healthy = client.post(
            "/v1/audio/speech",
            json={
                "model": DEFAULT_MODEL_ID,
                "input": "healthy",
                "voice": "aiden",
                "response_format": "pcm",
                "do_sample": False,
                "max_new_tokens": 2,
            },
        )
        assert healthy.status_code == 200
        assert healthy.content == b"healthy\0healthy\0"
    finally:
        service.stop(timeout_s=2)


def test_http_wav_has_one_exact_header_and_same_pcm_payload() -> None:
    service, client = _client()
    try:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": DEFAULT_MODEL_ID,
                "input": "ab",
                "voice": "aiden",
                "response_format": "wav",
                "do_sample": False,
                "max_new_tokens": 2,
            },
        )
        assert response.status_code == 200
        assert response.content[:4] == b"RIFF"
        assert response.content[8:12] == b"WAVE"
        assert struct.unpack("<I", response.content[40:44])[0] == 4
        assert response.content[44:] == b"abab"

        streaming = client.post(
            "/v1/audio/speech",
            json={
                "model": DEFAULT_MODEL_ID,
                "input": "ab",
                "voice": "aiden",
                "response_format": "wav",
                "stream": True,
                "do_sample": False,
                "max_new_tokens": 2,
            },
        )
        assert len(streaming.content) == 48
        assert struct.unpack("<I", streaming.content[4:8])[0] == 0xFFFFFFFF
        assert struct.unpack("<I", streaming.content[40:44])[0] == 0xFFFFFFFF
        assert streaming.content[44:] == b"abab"
    finally:
        service.stop(timeout_s=2)


def test_websocket_rejects_missing_protocol_and_cancellation_is_local() -> None:
    service, client = _client()
    try:
        with client.websocket_connect("/v1/audio/speech/ws") as socket:
            with pytest.raises(WebSocketDisconnect) as disconnected:
                socket.receive_json()
            assert disconnected.value.code == 1002
        with client.websocket_connect(
            "/v1/audio/speech/ws",
            subprotocols=[WS_PROTOCOL],
        ) as socket:
            socket.receive_json()
            socket.send_json(
                {
                    "type": "request.start",
                    "model": DEFAULT_MODEL_ID,
                    "voice": "aiden",
                    "do_sample": False,
                    "max_new_tokens": 2,
                }
            )
            socket.receive_json()
            socket.send_json({"type": "input_text.append", "sequence": 0, "text": "cancel me "})
            started = socket.receive_json()
            assert started["type"] == "response.started"
            assert started["request_id"]
            append_ack = socket.receive_json()
            assert append_ack["type"] == "input_text.ack"
            assert append_ack["sequence"] == 0
            socket.send_json({"type": "response.cancel"})
            done = socket.receive_json()
            assert done["type"] == "response.done"
            assert done["stop_reason"] == "cancelled"
        assert service.wait_idle(timeout_s=1)
        assert service.readiness().ready
    finally:
        service.stop(timeout_s=2)


def test_websocket_requires_protocol_and_streams_json_plus_binary() -> None:
    service, client = _client()
    try:
        with client.websocket_connect(
            "/v1/audio/speech/ws",
            subprotocols=[WS_PROTOCOL],
        ) as socket:
            assert socket.accepted_subprotocol == WS_PROTOCOL
            assert socket.receive_json()["type"] == "session.created"
            socket.send_json(
                {
                    "type": "request.start",
                    "model": DEFAULT_MODEL_ID,
                    "voice": "aiden",
                    "language": "english",
                    "do_sample": False,
                    "max_new_tokens": 2,
                }
            )
            assert socket.receive_json()["type"] == "request.configured"
            socket.send_json({"type": "input_text.append", "sequence": 0, "text": "ab "})
            started = socket.receive_json()
            assert started["type"] == "response.started"
            append_ack = socket.receive_json()
            assert append_ack == {
                "type": "input_text.ack",
                "event": "input_text.append",
                "sequence": 0,
                "request_id": append_ack["request_id"],
            }
            assert append_ack["request_id"]
            socket.send_json({"type": "input_text.end", "sequence": 1})
            controls = []
            audio = []
            while not any(item["type"] == "response.done" for item in controls):
                message = socket.receive()
                if message.get("bytes") is not None:
                    audio.append(message["bytes"])
                elif message.get("text") is not None:
                    controls.append(json.loads(message["text"]))
            end_ack = next(
                item
                for item in controls
                if item["type"] == "input_text.ack" and item["event"] == "input_text.end"
            )
            assert end_ack == {
                "type": "input_text.ack",
                "event": "input_text.end",
                "sequence": 1,
                "request_id": append_ack["request_id"],
            }
            assert started["request_id"] == append_ack["request_id"]
            assert audio == [b"ab \x00", b"ab \x00"]
            done = next(item for item in controls if item["type"] == "response.done")
            assert done["type"] == "response.done"
            assert done["request_id"] == append_ack["request_id"]
    finally:
        service.stop(timeout_s=2)


def test_health_reports_a_failed_owner_as_not_alive() -> None:
    """Liveness must let an orchestrator replace a process that cannot recover."""

    service = FakeEngine(FakeEngineBackend(), config=engine_config(max_buffered_bytes=128))
    service.start(request("probe"), timeout_s=2)
    try:
        with TestClient(create_app(service, text_frontend=service.model.text)) as client:
            alive = client.get("/health")
            assert alive.status_code == 200
            assert alive.json()["alive"] is True

            service._phase = ServicePhase.FAILED
            dead = client.get("/health")
            assert dead.status_code == 503
            assert dead.json() == {"alive": False, "phase": "failed"}
    finally:
        service._phase = ServicePhase.READY
        service.stop(timeout_s=2)

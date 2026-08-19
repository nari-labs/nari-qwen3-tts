"""FastAPI application composition for HTTP and WebSocket synthesis."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

import anyio.to_thread
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse

from nari_qwen3_tts.api.http import error_response, stream_bytes
from nari_qwen3_tts.api.schemas import (
    SpeechRequestBody,
)
from nari_qwen3_tts.api.wav import SAMPLE_RATE, wav_header
from nari_qwen3_tts.api.websocket import WS_PROTOCOL, handle_speech_websocket
from nari_qwen3_tts.config import DEFAULT_MODEL_ID, ApiConfig
from nari_qwen3_tts.contract.health import ServicePhase
from nari_qwen3_tts.contract.request import TextFrontend
from nari_qwen3_tts.engine.engine import Engine

_PCM_MEDIA_TYPE = "audio/pcm"
_WAV_MEDIA_TYPE = "audio/wav"
# Sockets, tokenization hops, and the readiness endpoints also borrow worker
# threads, so the transport needs slack above the admission ceiling.
def create_app(  # noqa: PLR0915
    engine: Engine,
    *,
    text_frontend: TextFrontend,
    config: ApiConfig | None = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> FastAPI:
    api_config = config or ApiConfig()
    tokenizer_slots = asyncio.Semaphore(
        max(1, int(text_frontend.streaming_tokenizer_concurrency))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # `POST /v1/audio/speech` is a blocking handler that occupies a worker
        # thread for the life of the request. Left at the framework default the
        # transport would stall well below `max_active_requests`, so the two
        # ceilings are tied together here rather than drifting apart.
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(
            limiter.total_tokens,
            engine.config.max_active_requests + api_config.transport_thread_headroom,
        )
        yield

    app = FastAPI(title="Nari Qwen3-TTS", lifespan=lifespan)

    @app.get("/health")
    def health() -> JSONResponse:
        """Liveness: is the Engine thread still able to serve anything at all?

        Readiness lives at `/ready`. A failed Engine keeps answering requests
        with 503 forever, so reporting it alive would stop an orchestrator from
        replacing a process that can never recover.
        """
        status = engine.readiness()
        alive = status.phase not in {ServicePhase.FAILED, ServicePhase.STOPPED}
        return JSONResponse(
            {"alive": alive, "phase": status.phase.value},
            status_code=200 if alive else 503,
        )

    @app.get("/ready")
    def ready() -> JSONResponse:
        status = engine.readiness()
        payload = {
            "ready": status.ready,
            "phase": status.phase.value,
            "reason": status.reason,
            "ordinary_pcm_bytes": status.ordinary_pcm_bytes,
            "ordinary_retired": status.ordinary_retired,
        }
        return JSONResponse(payload, status_code=200 if status.ready else 503)

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "qwen"}],
        }

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequestBody):
        request_id = f"http-{uuid.uuid4().hex}"
        try:
            # Body semantics the schema cannot express (unsupported model,
            # conflicting instruction aliases) are client errors. Rejecting them
            # here keeps `_error_response` free to treat a stray ValueError from
            # the serving stack as the internal fault it is.
            request = body.to_domain(served_model_id=model_id)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error
        try:
            stream = engine.submit(
                request_id,
                request,
                timeout_s=api_config.command_timeout_s,
            )
            media_type = _PCM_MEDIA_TYPE if body.response_format == "pcm" else _WAV_MEDIA_TYPE
            if body.stream:
                return StreamingResponse(
                    stream_bytes(
                        engine,
                        request_id,
                        stream,
                        response_format=body.response_format,
                        command_timeout_s=api_config.command_timeout_s,
                    ),
                    media_type=media_type,
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Nari-Request-ID": request_id,
                    },
                )
            pcm = b"".join(stream.iter_bytes(timeout_s=None))
            content = pcm if body.response_format == "pcm" else wav_header(pcm_bytes=len(pcm)) + pcm
            return Response(
                content,
                media_type=media_type,
                headers={"X-Nari-Request-ID": request_id},
            )
        except BaseException as error:
            raise error_response(error) from error

    @app.websocket("/v1/audio/speech/ws")
    async def speech_websocket_endpoint(websocket: WebSocket) -> None:
        await handle_speech_websocket(
            websocket,
            engine=engine,
            api_config=api_config,
            model_id=model_id,
            tokenizer=text_frontend,
            tokenizer_slots=tokenizer_slots,
        )

    return app


__all__ = [
    "SAMPLE_RATE",
    "SpeechRequestBody",
    "WS_PROTOCOL",
    "create_app",
    "wav_header",
]

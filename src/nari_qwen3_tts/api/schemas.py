"""Strict public HTTP and WebSocket request schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nari_qwen3_tts.config import DEFAULT_MODEL_ID
from nari_qwen3_tts.contract.request import SynthesisRequest


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _SynthesisControls(_StrictModel):
    """Generation controls shared by HTTP and WebSocket entry points."""

    model: str | None = None
    voice: str = "aiden"
    language: str = "auto"
    seed: Annotated[int, Field(ge=0)] = 0
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: Annotated[int, Field(ge=1, le=32 * 1024)] = 2048
    ignore_eos: bool = False
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    skip_fixed_bootstrap_audio: bool = True
    stream_chunk_schedule: Annotated[
        list[Annotated[int, Field(ge=1, le=128)]],
        Field(min_length=1, max_length=32),
    ] | None = None
    stream_chunk_frames: Annotated[int, Field(ge=1, le=128)] | None = None
    stream_first_chunk_frames: Annotated[int, Field(ge=1, le=128)] | None = None
    stream_steady_chunk_frames: Annotated[int, Field(ge=1, le=128)] | None = None

    def _controls(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _SynthesisControls.model_fields}


class SpeechRequestBody(_SynthesisControls):
    input: str
    instruct: str | None = None
    instructions: str | None = None
    response_format: Literal["pcm", "wav"] = "wav"
    stream: bool = False
    speed: float = 1.0
    non_streaming_mode: bool = True

    def to_domain(self, *, served_model_id: str = DEFAULT_MODEL_ID) -> SynthesisRequest:
        if self.model is not None and self.model != served_model_id:
            raise ValueError(f"unsupported model: {self.model!r}")
        if self.speed != 1.0:
            raise ValueError("Qwen3-TTS speed is fixed at 1.0")
        if (
            self.instruct is not None
            and self.instructions is not None
            and self.instruct != self.instructions
        ):
            raise ValueError("instruct and instructions cannot disagree")
        controls = self._controls()
        controls.pop("model")
        controls["random_seed"] = controls.pop("seed")
        return SynthesisRequest(
            text=self.input,
            instruct=self.instruct if self.instruct is not None else (self.instructions or ""),
            non_streaming_mode=self.non_streaming_mode,
            **controls,
        )


class WebSocketStart(_SynthesisControls):
    type: Literal["request.start"]
    instruction: str | None = None
    instruct: str | None = None
    instructions: str | None = None
    response_format: Literal["pcm", "pcm16", "pcm_s16le"] = "pcm"

    @model_validator(mode="after")
    def _one_instruction_alias(self):
        aliases = tuple(
            value
            for value in (self.instruction, self.instruct, self.instructions)
            if value is not None
        )
        if len(aliases) > 1:
            raise ValueError("set only one of instruction, instruct, or instructions")
        return self

    def to_domain(self, *, served_model_id: str = DEFAULT_MODEL_ID) -> SynthesisRequest:
        return SpeechRequestBody(
            input="live input pending",
            instruct=self.instruction or self.instruct or self.instructions,
            response_format="pcm",
            non_streaming_mode=False,
            **self._controls(),
        ).to_domain(served_model_id=served_model_id)


class WebSocketAppend(_StrictModel):
    type: Literal["input_text.append"]
    sequence: Annotated[int, Field(ge=0)]
    text: Annotated[str, Field(min_length=1)]


class WebSocketEnd(_StrictModel):
    type: Literal["input_text.end"]
    sequence: Annotated[int, Field(ge=0)]


class WebSocketCancel(_StrictModel):
    type: Literal["response.cancel"]


__all__ = [
    "SpeechRequestBody",
    "WebSocketAppend",
    "WebSocketCancel",
    "WebSocketEnd",
    "WebSocketStart",
]

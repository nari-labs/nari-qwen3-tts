"""Engine-owned mutable request state and bounded claim accounting."""

from __future__ import annotations

import math
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum

import torch

from nari_qwen3_tts.contract.codec_state import IncrementalCodecState
from nari_qwen3_tts.contract.frames import COLD_TERMINAL_PAD_FRAMES
from nari_qwen3_tts.contract.request import AdmittedRequest, TextContinuation
from nari_qwen3_tts.contract.rng import logical_rng_offset
from nari_qwen3_tts.contract.stage import SynthesisStage
from nari_qwen3_tts.contract.work import CodecBatchCompatibility

# One Talker draw per frame, at codebook 0. The Code Predictor owns codebooks
# 1..15 of the same frame, so Talker advances by one complete logical frame.
TALKER_RNG_FRAME_STRIDE = logical_rng_offset(1, 0)


class GenerationPhase(str, Enum):
    TALKER_PREFILL = "talker_prefill"
    CODE_PREDICTOR = "code_predictor"
    TALKER_DECODE = "talker_decode"
    DONE = "done"


class CodecPhase(str, Enum):
    COLLECTING = "collecting"
    READY = "ready"
    DONE = "done"


@dataclass(slots=True)
class GenerationLane:
    phase: GenerationPhase = GenerationPhase.TALKER_PREFILL
    generation_step: int = 0
    frame_index: int = 0
    next_sampling_offset: int = 0
    seen_token_mask: torch.Tensor | None = None
    token: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    step_input: torch.Tensor | None = None
    claim_token: int | None = None
    claim_batch_id: int | None = None
    version: int = 0


@dataclass(slots=True)
class CodecLane:
    phase: CodecPhase = CodecPhase.COLLECTING
    chunk_schedule: tuple[int, ...] = (2, 4, 8, 12)
    terminal_pad_frames: tuple[int, ...] = (12,)
    cold_terminal_pad_frames: tuple[int, ...] = COLD_TERMINAL_PAD_FRAMES
    suppress_first_silent_frame: bool = False
    buffered_frames: tuple[torch.Tensor, ...] = ()
    history_frames: tuple[torch.Tensor, ...] = ()
    context_frames_consumed: int = 0
    chunk_index: int = 0
    producer_done: bool = False
    decoder_state: IncrementalCodecState | None = None
    ready_compatibility: CodecBatchCompatibility | None = None
    in_flight_compatibility: CodecBatchCompatibility | None = None
    compute_terminal: bool = False
    pending_outputs: int = 0
    output_tracking: bool = False
    output_terminal: bool = False
    visible_pcm_frames: int = 0
    playback_started_at_s: float | None = None
    emitted_duration_s: float = 0.0
    last_routed_at_s: float | None = None
    execution_reserve_s: float = 0.0
    claim_token: int | None = None
    claim_batch_id: int | None = None
    version: int = 0

    def consume(self, frames: int) -> None:
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise ValueError("Codec consumed frames must be a non-negative integer")
        if frames > len(self.buffered_frames):
            raise ValueError("Codec consumed frame count exceeds the buffered prefix")
        self.buffered_frames = self.buffered_frames[frames:]
        self.context_frames_consumed += frames
        self.chunk_index += 1

    def mark_routed(self, pcm_bytes: int, *, at_s: float, sample_rate: int) -> None:
        if isinstance(pcm_bytes, bool) or not isinstance(pcm_bytes, int) or pcm_bytes < 0:
            raise ValueError("routed PCM byte count must be a non-negative integer")
        if pcm_bytes % 2:
            raise ValueError("routed PCM16 must contain whole samples")
        if (
            isinstance(at_s, bool)
            or not isinstance(at_s, (int, float))
            or not math.isfinite(at_s)
        ):
            raise ValueError("PCM route time must be finite")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
            raise ValueError("sample rate must be a positive integer")
        at_s = float(at_s)
        if self.last_routed_at_s is not None and at_s < self.last_routed_at_s:
            raise ValueError("PCM route time must be monotonic")
        self.last_routed_at_s = at_s
        if pcm_bytes:
            if self.playback_started_at_s is None:
                self.playback_started_at_s = at_s
            self.emitted_duration_s += (pcm_bytes // 2) / sample_rate


@dataclass(slots=True)
class LiveInputState:
    next_engine_sequence: int = 0
    input_finished: bool = False
    committed_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.input_finished, bool):
            raise TypeError("live input lifecycle flag must be a boolean")
        for name in (
            "next_engine_sequence",
            "committed_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(slots=True)
class PendingLiveInput:
    """One unpublished successor continuation and its ordered receipts."""

    continuation: TextContinuation
    receipts: list[Future[None]]

    def __post_init__(self) -> None:
        if not isinstance(self.continuation, TextContinuation):
            raise TypeError("pending live input requires a TextContinuation")
        if not self.receipts or any(not isinstance(receipt, Future) for receipt in self.receipts):
            raise TypeError("pending live input requires Future receipts")


@dataclass(slots=True)
class RequestState:
    request_id: str
    input: AdmittedRequest | None = None
    generation: GenerationLane = field(default_factory=GenerationLane)
    codec: CodecLane = field(default_factory=CodecLane)
    cancel_requested: bool = False
    admission_sequence: int = 0
    pending_gpu_submissions: int = 0
    pending_live_input: PendingLiveInput | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if self.input is not None and not isinstance(self.input, AdmittedRequest):
            raise TypeError("request input must be an AdmittedRequest")
        for name in ("admission_sequence", "pending_gpu_submissions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def for_testing(cls, request_id: str) -> "RequestState":
        return cls(request_id=request_id)

    @property
    def is_removable(self) -> bool:
        lifetimes_clear = (
            self.generation.claim_token is None
            and self.generation.claim_batch_id is None
            and self.codec.claim_token is None
            and self.codec.claim_batch_id is None
            and self.pending_gpu_submissions == 0
            and self.codec.pending_outputs == 0
            and self.pending_live_input is None
        )
        if not lifetimes_clear:
            return False
        if self.cancel_requested:
            return True
        terminal = (
            self.generation.phase is GenerationPhase.DONE
            and self.codec.phase is CodecPhase.DONE
        )
        output_ready = not self.codec.output_tracking or self.codec.output_terminal
        return terminal and output_ready

    @property
    def version(self) -> int:
        return max(self.generation.version, self.codec.version)

    def version_for(self, stage: SynthesisStage) -> int:
        if not isinstance(stage, SynthesisStage):
            raise TypeError("request version requires a SynthesisStage")
        return self.codec.version if stage is SynthesisStage.CODEC else self.generation.version

    @staticmethod
    def _value_signature(value: object | None) -> object:
        if value is None:
            return None
        tolist = getattr(value, "tolist", None)
        return tolist() if callable(tolist) else value

    def committed_view(self) -> tuple[object, ...]:
        """Return request-visible progress only, excluding speculative claims."""
        return (
            self.version,
            self.cancel_requested,
            self.generation.phase,
            self.generation.generation_step,
            self.generation.frame_index,
            self.generation.next_sampling_offset,
            self._value_signature(self.generation.seen_token_mask),
            self._value_signature(self.generation.token),
            self._value_signature(self.generation.hidden),
            self._value_signature(self.generation.logits),
            self._value_signature(self.generation.step_input),
            self.codec.phase,
            self.codec.version,
            self.codec.chunk_index,
            self.codec.context_frames_consumed,
            tuple(self._value_signature(value) for value in self.codec.buffered_frames),
            self.codec.pending_outputs,
            self.codec.output_terminal,
            self.codec.producer_done,
            self.codec.compute_terminal,
            self.codec.visible_pcm_frames,
            self.codec.playback_started_at_s,
            self.codec.emitted_duration_s,
            self.codec.last_routed_at_s,
        )


class EngineStateError(RuntimeError):
    pass


class StaleCompletionError(EngineStateError):
    pass


class DuplicateCompletionError(EngineStateError):
    pass


class ResourceExhaustedError(EngineStateError):
    pass


class RequestStateStore:
    """Engine-owned request collection and bounded in-flight row accounting."""

    def __init__(self, *, max_in_flight_rows: int = 1024) -> None:
        if isinstance(max_in_flight_rows, bool) or not isinstance(max_in_flight_rows, int):
            raise TypeError("max_in_flight_rows must be an integer")
        if max_in_flight_rows < 1:
            raise ValueError("max_in_flight_rows must be positive")
        self.max_in_flight_rows = max_in_flight_rows
        self._requests: dict[str, RequestState] = {}
        self._in_flight_rows = 0

    @property
    def in_flight_rows(self) -> int:
        return self._in_flight_rows

    @property
    def available_in_flight_rows(self) -> int:
        return self.max_in_flight_rows - self._in_flight_rows

    @property
    def requests(self) -> tuple[RequestState, ...]:
        return tuple(
            sorted(
                self._requests.values(),
                key=lambda state: (state.admission_sequence, state.request_id),
            )
        )

    def admit(self, state: RequestState) -> None:
        if state.request_id in self._requests:
            raise ValueError(f"request {state.request_id!r} is already admitted")
        self._requests[state.request_id] = state

    def request(self, request_id: str) -> RequestState:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"unknown Qwen3-TTS request {request_id!r}") from error

    def cancel(self, request_id: str) -> None:
        state = self.request(request_id)
        state.cancel_requested = True
        if state.generation.claim_token is None:
            state.generation.phase = GenerationPhase.DONE
        if state.codec.claim_token is None:
            state.codec.phase = CodecPhase.DONE

    def remove(self, request_id: str) -> None:
        state = self.request(request_id)
        if not state.is_removable:
            raise EngineStateError("request still owns live or in-flight state")
        del self._requests[request_id]


__all__ = [
    "TALKER_RNG_FRAME_STRIDE",
    "CodecLane",
    "CodecPhase",
    "DuplicateCompletionError",
    "EngineStateError",
    "GenerationLane",
    "GenerationPhase",
    "LiveInputState",
    "PendingLiveInput",
    "RequestState",
    "RequestStateStore",
    "ResourceExhaustedError",
    "StaleCompletionError",
]

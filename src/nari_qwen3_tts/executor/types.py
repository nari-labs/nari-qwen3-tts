"""Typed values crossing the fixed Qwen3-TTS model-job boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nari_qwen3_tts.executor.talker_kv import TalkerAttentionContext
from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState


def _require_vector(name: str, value: torch.Tensor, rows: int) -> None:
    if value.ndim != 1 or value.shape[0] != rows:
        raise ValueError(f"{name} row count must be {rows}, got {tuple(value.shape)}")


def _require_dtype(name: str, value: torch.Tensor, dtype: torch.dtype) -> None:
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")


def _require_floating(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _require_same_device(reference_name: str, reference: torch.Tensor, **values: torch.Tensor | None) -> None:
    mismatched = [name for name, value in values.items() if value is not None and value.device != reference.device]
    if mismatched:
        raise ValueError(f"{reference_name} and {', '.join(mismatched)} must share one device")


@dataclass(frozen=True, slots=True)
class TalkerSamplingInput:
    """Complete stateless sampling inputs in logical request-row order."""

    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    seed: torch.Tensor
    offsets: torch.Tensor
    seen_token_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        rows = int(self.temperature.shape[0]) if self.temperature.ndim == 1 else -1
        if rows < 0:
            raise ValueError("temperature must be a row vector")
        if rows == 0:
            raise ValueError("Talker sampling requires at least one row")
        for name in ("top_k", "top_p", "repetition_penalty", "seed", "offsets"):
            _require_vector(name, getattr(self, name), rows)
        for name in ("temperature", "top_p", "repetition_penalty"):
            _require_floating(name, getattr(self, name))
        _require_dtype("top_k", self.top_k, torch.int32)
        _require_dtype("seed", self.seed, torch.long)
        _require_dtype("offsets", self.offsets, torch.long)
        if self.seen_token_mask is not None:
            if self.seen_token_mask.ndim != 2 or self.seen_token_mask.shape[0] != rows:
                raise ValueError("seen_token_mask row count must match sampling rows")
            _require_dtype("seen_token_mask", self.seen_token_mask, torch.bool)
        _require_same_device(
            "temperature",
            self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            offsets=self.offsets,
            seen_token_mask=self.seen_token_mask,
        )

    @property
    def rows(self) -> int:
        return int(self.temperature.shape[0])


@dataclass(frozen=True, slots=True)
class TalkerPrefillInput:
    attention_context: TalkerAttentionContext
    text_token_ids: torch.Tensor
    codec_token_ids: torch.Tensor
    codec_token_mask: torch.Tensor
    last_token_indices: torch.Tensor
    suppress_eos: torch.Tensor
    sampling: TalkerSamplingInput

    def __post_init__(self) -> None:
        tokens = int(self.text_token_ids.shape[0]) if self.text_token_ids.ndim == 1 else -1
        if tokens < 0:
            raise ValueError("prefill text_token_ids must be packed into one vector")
        if tokens == 0:
            raise ValueError("Talker prefill requires at least one packed token")
        _require_dtype("text_token_ids", self.text_token_ids, torch.long)
        _require_dtype("codec_token_ids", self.codec_token_ids, torch.long)
        _require_dtype("codec_token_mask", self.codec_token_mask, torch.bool)
        for name in ("codec_token_ids", "codec_token_mask"):
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != tokens:
                raise ValueError(f"{name} must match the packed token count")
        rows = self.sampling.rows
        _require_vector("last_token_indices", self.last_token_indices, rows)
        _require_vector("suppress_eos", self.suppress_eos, rows)
        _require_dtype("last_token_indices", self.last_token_indices, torch.long)
        _require_dtype("suppress_eos", self.suppress_eos, torch.bool)
        if torch.any((self.last_token_indices < 0) | (self.last_token_indices >= tokens)):
            raise ValueError("last_token_indices must address the packed Talker input")
        _require_same_device(
            "text_token_ids",
            self.text_token_ids,
            codec_token_ids=self.codec_token_ids,
            codec_token_mask=self.codec_token_mask,
            last_token_indices=self.last_token_indices,
            suppress_eos=self.suppress_eos,
            sampling_temperature=self.sampling.temperature,
        )


@dataclass(frozen=True, slots=True)
class TalkerDecodeInput:
    attention_context: TalkerAttentionContext
    talker_step_embed: torch.Tensor
    text_token_ids: torch.Tensor
    suppress_eos: torch.Tensor
    sampling: TalkerSamplingInput

    def __post_init__(self) -> None:
        rows = self.sampling.rows
        if self.talker_step_embed.ndim != 2 or self.talker_step_embed.shape[0] != rows:
            raise ValueError("talker_step_embed row count must match sampling rows")
        _require_vector("text_token_ids", self.text_token_ids, rows)
        _require_vector("suppress_eos", self.suppress_eos, rows)
        _require_floating("talker_step_embed", self.talker_step_embed)
        _require_dtype("text_token_ids", self.text_token_ids, torch.long)
        _require_dtype("suppress_eos", self.suppress_eos, torch.bool)
        _require_same_device(
            "talker_step_embed",
            self.talker_step_embed,
            text_token_ids=self.text_token_ids,
            suppress_eos=self.suppress_eos,
            sampling_temperature=self.sampling.temperature,
        )


@dataclass(frozen=True, slots=True)
class TalkerResult:
    tokens: torch.Tensor
    last_hidden: torch.Tensor
    logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 1 or self.tokens.shape[0] < 1:
            raise ValueError("Talker result tokens must be a non-empty row vector")
        _require_dtype("Talker result tokens", self.tokens, torch.long)
        rows = int(self.tokens.shape[0])
        if self.last_hidden.ndim != 2 or self.last_hidden.shape[0] != rows:
            raise ValueError("Talker result last_hidden row count must match tokens")
        if self.logits.ndim != 2 or self.logits.shape[0] != rows:
            raise ValueError("Talker result logits row count must match tokens")
        _require_floating("Talker result last_hidden", self.last_hidden)
        _require_floating("Talker result logits", self.logits)
        _require_same_device(
            "Talker result tokens",
            self.tokens,
            last_hidden=self.last_hidden,
            logits=self.logits,
        )


@dataclass(frozen=True, slots=True)
class CodePredictorInput:
    layer0_token: torch.Tensor
    past_hidden: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    seed: torch.Tensor
    offsets: torch.Tensor
    num_code_groups: int
    position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.num_code_groups, bool) or not isinstance(self.num_code_groups, int):
            raise TypeError("num_code_groups must be an integer")
        if self.num_code_groups < 2:
            raise ValueError("num_code_groups must include layer zero and at least one residual group")
        rows = int(self.layer0_token.shape[0]) if self.layer0_token.ndim == 1 else -1
        if rows < 0:
            raise ValueError("layer0_token must be a row vector")
        if rows == 0:
            raise ValueError("Code Predictor input requires at least one row")
        _require_dtype("layer0_token", self.layer0_token, torch.long)
        if self.past_hidden.ndim != 2 or self.past_hidden.shape[0] != rows:
            raise ValueError("past_hidden row count must match layer0_token")
        _require_floating("past_hidden", self.past_hidden)
        for name in ("temperature", "top_k", "top_p", "seed"):
            _require_vector(name, getattr(self, name), rows)
        _require_floating("temperature", self.temperature)
        _require_dtype("top_k", self.top_k, torch.int32)
        _require_floating("top_p", self.top_p)
        _require_dtype("seed", self.seed, torch.long)
        expected = (rows, self.num_code_groups - 1)
        if tuple(self.offsets.shape) != expected:
            raise ValueError(f"offsets must cover all residual groups with shape {expected}")
        _require_dtype("offsets", self.offsets, torch.long)
        if self.position_ids is not None and tuple(self.position_ids.shape) != (rows, self.num_code_groups):
            raise ValueError("position_ids must cover every code group and row")
        if self.position_ids is not None:
            _require_dtype("position_ids", self.position_ids, torch.long)
        _require_same_device(
            "layer0_token",
            self.layer0_token,
            past_hidden=self.past_hidden,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            seed=self.seed,
            offsets=self.offsets,
            position_ids=self.position_ids,
        )


@dataclass(frozen=True, slots=True)
class CodePredictorResult:
    frames: torch.Tensor
    codec_sum: torch.Tensor

    def __post_init__(self) -> None:
        if self.frames.ndim != 2 or self.frames.shape[0] < 1 or self.frames.shape[1] < 2:
            raise ValueError("Code Predictor result frames must contain rows and complete code groups")
        _require_dtype("Code Predictor result frames", self.frames, torch.long)
        if self.codec_sum.ndim != 2 or self.codec_sum.shape[0] != self.frames.shape[0]:
            raise ValueError("Code Predictor result codec_sum row count must match frames")
        _require_floating("Code Predictor result codec_sum", self.codec_sum)
        _require_same_device(
            "Code Predictor result frames",
            self.frames,
            codec_sum=self.codec_sum,
        )


@dataclass(frozen=True, slots=True)
class CodecResult:
    pcm: torch.Tensor
    states: tuple[IncrementalCodecState, ...] | None
    terminal: bool
    pcm_lengths: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.pcm.ndim != 2 or self.pcm.shape[0] < 1:
            raise ValueError("Codec PCM result must have a non-empty batch dimension")
        if self.pcm.dtype != torch.int16:
            raise TypeError("Codec PCM result must use torch.int16")
        if not isinstance(self.terminal, bool):
            raise TypeError("Codec terminal flag must be a boolean")
        rows, width = self.pcm.shape
        if self.states is not None:
            if not isinstance(self.states, tuple):
                raise TypeError("Codec result states must be an owned tuple")
            if any(not isinstance(state, IncrementalCodecState) for state in self.states):
                raise TypeError("Codec result states must contain IncrementalCodecState values")
            if len(self.states) != rows:
                raise ValueError("Codec result state row count must match PCM")
        if self.pcm_lengths is not None:
            if not isinstance(self.pcm_lengths, tuple):
                raise TypeError("Codec PCM lengths must be an owned tuple")
            if len(self.pcm_lengths) != rows:
                raise ValueError("Codec PCM length row count must match PCM")
            if any(
                isinstance(length, bool)
                or not isinstance(length, int)
                or length < 0
                or length > width
                for length in self.pcm_lengths
            ):
                raise ValueError("Codec PCM lengths must be integers within the PCM width")

    def pcm_row(self, row: int) -> torch.Tensor:
        if self.pcm.ndim != 2:
            raise ValueError("Codec PCM result must have a batch dimension")
        length = self.pcm.shape[1] if self.pcm_lengths is None else self.pcm_lengths[row]
        return self.pcm[row, :length]


__all__ = [
    "CodePredictorInput",
    "CodePredictorResult",
    "CodecResult",
    "TalkerDecodeInput",
    "TalkerPrefillInput",
    "TalkerResult",
    "TalkerSamplingInput",
]

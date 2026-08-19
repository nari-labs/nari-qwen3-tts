"""Finite config-derived capture lowering owned by the synthesis planner."""

from __future__ import annotations

from dataclasses import dataclass

from nari_qwen3_tts.contract.frames import WARM_TEMPLATE_FRAMES, WHOLE_SEQUENCE_MAX_FRAMES
from nari_qwen3_tts.contract.stage import (
    CodecCaptureKey,
    CodecExecutionMode,
    CodePredictorCaptureKey,
    CudaGraphKey,
    TalkerDecodeCaptureKey,
    TalkerPrefillCaptureKey,
)
from nari_qwen3_tts.profile import StageExecutionConfig


class CaptureCoverageError(RuntimeError):
    """Raised instead of executing an uncaptured non-empty shape eagerly."""



@dataclass(frozen=True, slots=True)
class PhysicalBatchSlice:
    key: CudaGraphKey | None
    logical_start: int
    logical_stop: int
    capture_batch_size: int
    padding: int
    metadata_only: bool = False

    @property
    def logical_rows(self) -> int:
        return self.logical_stop - self.logical_start


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _lower_rows(
    logical_rows: int,
    sizes: tuple[int, ...],
    key_factory,
) -> tuple[PhysicalBatchSlice, ...]:
    _require_positive_integer("logical batch size", logical_rows)
    if not sizes:
        raise CaptureCoverageError("CUDA capture batch catalog is empty")
    parts: list[PhysicalBatchSlice] = []
    start = 0
    maximum = sizes[-1]
    while logical_rows - start > maximum:
        stop = start + maximum
        parts.append(PhysicalBatchSlice(key_factory(maximum), start, stop, maximum, 0))
        start = stop
    remaining = logical_rows - start
    capture_rows = next((size for size in sizes if size >= remaining), None)
    if capture_rows is None:
        raise CaptureCoverageError(f"no capture can serve {remaining} logical rows")
    parts.append(
        PhysicalBatchSlice(
            key_factory(capture_rows),
            start,
            logical_rows,
            capture_rows,
            capture_rows - remaining,
        )
    )
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class CaptureCatalog:
    talker_prefill: frozenset[TalkerPrefillCaptureKey]
    talker_decode: frozenset[TalkerDecodeCaptureKey]
    code_predictor: frozenset[CodePredictorCaptureKey]
    codec: frozenset[CodecCaptureKey]
    terminal_pad_frames: tuple[int, ...]
    prefill_max_batch_size: int
    codec_max_batch_size: int
    codec_whole_sequence_max_batch_size: int

    @classmethod
    def from_config(cls, config: StageExecutionConfig) -> CaptureCatalog:
        prefill = config.talker_prefill
        exact = {
            TalkerPrefillCaptureKey(batch, batch * sequence, sequence)
            for sequence in prefill.exact_sequence_lengths
            for batch in prefill.exact_batch_sizes
        }
        bucketed = {
            TalkerPrefillCaptureKey(batch, tokens, None)
            for batch in prefill.batch_sizes
            for tokens in prefill.token_buckets
        }
        codec_config = config.codec
        batches = codec_config.batches
        frames = codec_config.frames
        codec = {
            CodecCaptureKey(CodecExecutionMode.WHOLE_SEQUENCE, 1, batch) for batch in batches.whole_sequence_first_frame
        }
        codec.update(
            CodecCaptureKey(CodecExecutionMode.WHOLE_SEQUENCE, frame_count, batch)
            for frame_count in range(2, WHOLE_SEQUENCE_MAX_FRAMES + 1)
            for batch in batches.whole_sequence_followup
        )
        partial_cold_frames = set(frames.cold_terminal_partial)
        codec.update(
            CodecCaptureKey(CodecExecutionMode.COLD, frame_count, batch)
            for frame_count in frames.cold
            for batch in (
                batches.cold_terminal_partial
                if frame_count in partial_cold_frames
                else batches.cold
            )
        )
        full_frames = set(frames.warm_full)
        codec.update(
            CodecCaptureKey(CodecExecutionMode.WARM, frame_count, batch)
            for frame_count in frames.warm
            for batch in (batches.warm_full if frame_count in full_frames else batches.warm_partial)
        )
        return cls(
            talker_prefill=frozenset(exact | bucketed),
            talker_decode=frozenset(TalkerDecodeCaptureKey(size) for size in config.talker_decode.batch_sizes),
            code_predictor=frozenset(
                CodePredictorCaptureKey(size) for size in config.code_predictor.batch_sizes
            ),
            codec=frozenset(codec),
            terminal_pad_frames=frames.terminal_pad,
            prefill_max_batch_size=prefill.max_batch_size,
            codec_max_batch_size=codec_config.max_batch_size,
            codec_whole_sequence_max_batch_size=min(
                max(batches.whole_sequence_followup),
                codec_config.max_batch_size,
            ),
        )

    @property
    def required_keys(self) -> frozenset[CudaGraphKey]:
        return frozenset((*self.talker_prefill, *self.talker_decode, *self.code_predictor, *self.codec))

    @property
    def cold_terminal_pad_frames(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    key.model_frames
                    for key in self.codec
                    if key.mode is CodecExecutionMode.COLD
                }
            )
        )

    def _require_codec_frames(
        self,
        mode: CodecExecutionMode,
        first: int,
        last: int,
    ) -> None:
        if first > last:
            return
        captured = {
            key.model_frames
            for key in self.codec
            if key.mode is mode
        }
        present = sum(first <= frames <= last for frames in captured)
        required = last - first + 1
        if present == required:
            return
        search_stop = min(last, first + len(captured))
        missing = next(
            frames for frames in range(first, search_stop + 1) if frames not in captured
        )
        raise CaptureCoverageError(
            f"no {mode.value} Codec capture exists for {missing} frames"
        )

    def _require_warm_chunk(self, frames: int) -> None:
        captured = {
            key.model_frames
            for key in self.codec
            if key.mode is CodecExecutionMode.WARM
        }
        if frames not in captured:
            raise CaptureCoverageError(f"no warm Codec capture exists for {frames} frames")
        previous = 0
        for pad in self.terminal_pad_frames:
            if frames > previous and pad not in captured:
                raise CaptureCoverageError(
                    f"no warm Codec capture exists for terminal pad {pad} frames"
                )
            if frames <= pad:
                return
            previous = pad
        self._require_codec_frames(CodecExecutionMode.WARM, previous + 1, frames)

    def validate_codec_schedule(self, chunk_schedule: tuple[int, ...]) -> None:
        """Prove every cold, warm, and terminal capture before admission."""

        if not chunk_schedule or any(
            isinstance(frames, bool) or not isinstance(frames, int) or frames < 1
            for frames in chunk_schedule
        ):
            raise ValueError("Codec chunk schedule must contain positive integers")
        context_frames = 0
        chunk_index = 0
        while True:
            chunk_frames = chunk_schedule[min(chunk_index, len(chunk_schedule) - 1)]
            total_frames = context_frames + chunk_frames
            whole_sequence_frames = min(
                WHOLE_SEQUENCE_MAX_FRAMES,
                total_frames,
            )
            self._require_codec_frames(
                CodecExecutionMode.WHOLE_SEQUENCE,
                context_frames + 1,
                whole_sequence_frames,
            )
            if total_frames > WHOLE_SEQUENCE_MAX_FRAMES and total_frames not in self.cold_terminal_pad_frames:
                raise CaptureCoverageError(
                    f"no cold Codec capture exists for {total_frames} frames"
                )
            context_frames = total_frames
            chunk_index += 1
            if context_frames > WHOLE_SEQUENCE_MAX_FRAMES:
                break
        warm_chunks = (*chunk_schedule[chunk_index:], chunk_schedule[-1])
        for chunk_frames in dict.fromkeys(warm_chunks):
            self._require_warm_chunk(chunk_frames)

    def codec_batch_capacity(
        self,
        mode: CodecExecutionMode,
        *,
        model_frames: int,
    ) -> int:
        if not isinstance(mode, CodecExecutionMode):
            raise TypeError("Codec mode must be a CodecExecutionMode")
        if model_frames == 0:
            return self.codec_max_batch_size
        _require_positive_integer("Codec model frames", model_frames)
        if mode is CodecExecutionMode.WHOLE_SEQUENCE:
            # Match Qwen3TTSCodecSubmodule.batch_scheduling_hint: the deadline-aware
            # phase cap is the whole-sequence followup surface even though the
            # one-frame acceleration capture itself extends to larger rows.
            return self.codec_whole_sequence_max_batch_size
        sizes = tuple(
            key.capture_batch_size
            for key in self.codec
            if key.mode is mode and key.model_frames == model_frames
        )
        if not sizes:
            raise CaptureCoverageError(
                f"no {mode.value} Codec capture exists for {model_frames} frames"
            )
        return min(max(sizes), self.codec_max_batch_size)

    def lower_talker_prefill(
        self,
        sequence_lengths: tuple[int, ...],
    ) -> tuple[PhysicalBatchSlice, ...]:
        if not isinstance(sequence_lengths, tuple) or not sequence_lengths:
            raise ValueError("Talker prefill sequence lengths must be a non-empty tuple")
        if any(isinstance(length, bool) or not isinstance(length, int) or length <= 0 for length in sequence_lengths):
            raise ValueError("Talker prefill sequence lengths must be positive integers")
        parts: list[PhysicalBatchSlice] = []
        maximum_tokens = max(key.token_capacity for key in self.talker_prefill)
        start = 0
        while start < len(sequence_lengths):
            if sequence_lengths[start] > maximum_tokens:
                raise CaptureCoverageError("Talker prefill capture token capacity exceeded")
            stop = start
            tokens = 0
            while stop < len(sequence_lengths) and stop - start < self.prefill_max_batch_size:
                proposed = tokens + sequence_lengths[stop]
                if proposed > maximum_tokens:
                    break
                tokens = proposed
                stop += 1
            lengths = sequence_lengths[start:stop]
            rows = len(lengths)
            candidates = [
                key
                for key in self.talker_prefill
                if key.capture_batch_size >= rows and key.token_capacity >= tokens
            ]
            if not candidates:
                raise CaptureCoverageError(
                    f"no Talker prefill capture covers {rows} rows and {tokens} tokens"
                )
            key = min(
                candidates,
                key=lambda item: (
                    item.token_capacity,
                    item.capture_batch_size,
                    item.capture_sequence_length is None,
                    item.capture_sequence_length or 0,
                ),
            )
            parts.append(
                PhysicalBatchSlice(
                    key,
                    start,
                    start + rows,
                    key.capture_batch_size,
                    key.capture_batch_size - rows,
                )
            )
            start = stop
        return tuple(parts)

    def lower_talker_decode(self, logical_rows: int) -> tuple[PhysicalBatchSlice, ...]:
        sizes = tuple(sorted(key.capture_batch_size for key in self.talker_decode))
        return _lower_rows(logical_rows, sizes, TalkerDecodeCaptureKey)

    def lower_code_predictor(self, logical_rows: int) -> tuple[PhysicalBatchSlice, ...]:
        sizes = tuple(sorted(key.capture_batch_size for key in self.code_predictor))
        return _lower_rows(logical_rows, sizes, CodePredictorCaptureKey)

    def lower_codec(
        self,
        mode: CodecExecutionMode,
        *,
        model_frames: int,
        logical_rows: int,
    ) -> tuple[PhysicalBatchSlice, ...]:
        sizes = tuple(
            sorted(
                key.capture_batch_size
                for key in self.codec
                if key.mode is mode and key.model_frames == model_frames
            )
        )
        if not sizes:
            raise CaptureCoverageError(f"no {mode.value} Codec capture exists for {model_frames} frames")
        return _lower_rows(
            logical_rows,
            sizes,
            lambda rows: CodecCaptureKey(mode, model_frames, rows),
        )

    @staticmethod
    def lower_empty_terminal(logical_rows: int) -> PhysicalBatchSlice:
        _require_positive_integer("logical batch size", logical_rows)
        return PhysicalBatchSlice(None, 0, logical_rows, 0, 0, metadata_only=True)


__all__ = [
    "WHOLE_SEQUENCE_MAX_FRAMES",
    "WARM_TEMPLATE_FRAMES",
    "CaptureCoverageError",
    "CodePredictorCaptureKey",
    "CodecCaptureKey",
    "CodecExecutionMode",
    "CudaGraphKey",
    "CaptureCatalog",
    "PhysicalBatchSlice",
    "TalkerDecodeCaptureKey",
    "TalkerPrefillCaptureKey",
]

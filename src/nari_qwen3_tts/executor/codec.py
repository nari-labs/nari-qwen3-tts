"""Typed static-buffer CUDA executor for the fixed Codec lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nari_qwen3_tts.contract.frames import WARM_TEMPLATE_FRAMES
from nari_qwen3_tts.contract.stage import CodecCaptureKey, CodecExecutionMode
from nari_qwen3_tts.executor.cuda_graph import (
    CapturedCall,
    CaptureDriver,
    CudaGraphPoolFence,
    SlotLeaseState,
    TorchCaptureDriver,
)
from nari_qwen3_tts.executor.rows import (
    CodecRowsExecutionInput,
)
from nari_qwen3_tts.executor.types import CodecResult
from nari_qwen3_tts.model.incremental_codec import (
    IncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)

_STATE_MAPPINGS = (
    "transformer_keys",
    "transformer_values",
    "conv_histories",
    "transconv_overlaps",
)


@dataclass(slots=True)
class _StateBuffer:
    mapping: str
    key: int | str
    values: torch.Tensor


@dataclass(slots=True)
class _CodecCaptureState:
    inputs: tuple[_StateBuffer, ...]
    base_position_ids: torch.Tensor
    metadata: torch.Tensor
    host_metadata: torch.Tensor
    outputs: tuple[IncrementalCodecState, ...] = ()


@dataclass(slots=True)
class _CodecSlot:
    frames: torch.Tensor
    state: _CodecCaptureState | None
    call: CapturedCall
    lease_state: SlotLeaseState


class CodecExecutor:
    """Capture whole-sequence, coherent-cold, and coherent-warm Codec shapes separately."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        num_code_groups: int,
        cold_frame_sizes: tuple[int, ...],
        device: torch.device,
        incremental_decoder: object | None = None,
        driver: CaptureDriver | None = None,
        submission_fence: CudaGraphPoolFence | None = None,
    ) -> None:
        if isinstance(num_code_groups, bool) or not isinstance(num_code_groups, int):
            raise TypeError("Codec num_code_groups must be an integer")
        if num_code_groups < 2:
            raise ValueError("Codec num_code_groups must include layer zero and residual groups")
        if WARM_TEMPLATE_FRAMES not in cold_frame_sizes:
            raise ValueError(
                f"warm Codec state is templated from a {WARM_TEMPLATE_FRAMES}-frame cold decode, "
                f"which is not among the captured cold shapes {tuple(cold_frame_sizes)}"
            )
        self.model = model
        self.num_code_groups = num_code_groups
        self.incremental_decoder = incremental_decoder or Qwen3TTSIncrementalDecoder(
            model.decoder
        )
        self.samples_per_frame = int(self.incremental_decoder.samples_per_frame)
        if self.samples_per_frame < 1:
            raise ValueError("Codec samples_per_frame must be positive")
        self.retained_context = self._resolve_retained_context(self.incremental_decoder)
        self.cold_frame_sizes = tuple(cold_frame_sizes)
        self.device = device
        self.driver = driver or TorchCaptureDriver(device=device, autocast_dtype=None)
        self.submission_fence = submission_fence or CudaGraphPoolFence(device=device)
        self._slots: dict[CodecCaptureKey, _CodecSlot] = {}
        self._warm_template: IncrementalCodecState | None = None

    @staticmethod
    def _resolve_retained_context(incremental_decoder: object) -> int:
        retained_context = getattr(incremental_decoder, "retained_context", None)
        if retained_context is None:
            decoder = getattr(incremental_decoder, "decoder", None)
            transformer = getattr(decoder, "pre_transformer", None)
            config = getattr(transformer, "config", None)
            sliding_window = getattr(config, "sliding_window", None)
            if sliding_window is None:
                raise TypeError(
                    "Codec decoder exposes neither retained_context nor "
                    "decoder.pre_transformer.config.sliding_window"
                )
            retained_context = max(0, int(sliding_window) - 1)
        if (
            isinstance(retained_context, bool)
            or not isinstance(retained_context, int)
            or retained_context < 0
        ):
            raise ValueError("Codec retained_context must be a non-negative integer")
        return retained_context

    @staticmethod
    def new_state() -> IncrementalCodecState:
        return IncrementalCodecState()

    @staticmethod
    def clone_state(state: IncrementalCodecState) -> IncrementalCodecState:
        clone = IncrementalCodecState(
            frame_position=state.frame_position,
            transformer_context_length=state.transformer_context_length,
        )
        for name in _STATE_MAPPINGS:
            setattr(
                clone,
                name,
                {
                    key: value.detach().clone()
                    for key, value in getattr(state, name).items()
                },
            )
        return clone

    @staticmethod
    def _mapping_signature(
        state: IncrementalCodecState,
    ) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
        signature: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
        for name in _STATE_MAPPINGS:
            mapping = getattr(state, name)
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(
                    "incremental Codec state mappings must be non-empty dictionaries"
                )
            entries: list[tuple[object, ...]] = []
            for key, value in sorted(mapping.items(), key=lambda item: str(item[0])):
                if not isinstance(value, torch.Tensor) or value.numel() == 0:
                    raise ValueError(
                        "incremental Codec state mappings must contain non-empty tensors"
                    )
                entries.append((key, tuple(value.shape), value.dtype, value.device))
            signature.append((name, tuple(entries)))
        if state.transformer_keys.keys() != state.transformer_values.keys():
            raise ValueError(
                "incremental Codec transformer key/value mappings must have identical layers"
            )
        for key in state.transformer_keys:
            key_tensor = state.transformer_keys[key]
            value_tensor = state.transformer_values[key]
            if (
                key_tensor.shape != value_tensor.shape
                or key_tensor.dtype != value_tensor.dtype
                or key_tensor.device != value_tensor.device
            ):
                raise ValueError(
                    "incremental Codec transformer key/value tensors must have identical structure"
                )
        return tuple(signature)

    @classmethod
    def _require_cold_states(
        cls,
        states: list[IncrementalCodecState] | tuple[IncrementalCodecState, ...],
    ) -> None:
        if not states:
            raise ValueError("Codec jobs require at least one row")
        for state in states:
            mappings_empty = all(
                isinstance(getattr(state, name), dict) and not getattr(state, name)
                for name in _STATE_MAPPINGS
            )
            if (
                not isinstance(state.frame_position, int)
                or isinstance(state.frame_position, bool)
                or not isinstance(state.transformer_context_length, int)
                or isinstance(state.transformer_context_length, bool)
                or state.frame_position != 0
                or state.transformer_context_length != 0
                or not mappings_empty
            ):
                raise ValueError("state bootstrap requires coherent cold Codec states")

    @classmethod
    def _require_warm_states(
        cls,
        states: list[IncrementalCodecState] | tuple[IncrementalCodecState, ...],
        *,
        retained_context: int,
    ) -> None:
        if not states:
            raise ValueError("Codec jobs require at least one row")
        expected_signature: tuple[
            tuple[str, tuple[tuple[object, ...], ...]], ...
        ] | None = None
        for state in states:
            if (
                not isinstance(state.frame_position, int)
                or isinstance(state.frame_position, bool)
                or not isinstance(state.transformer_context_length, int)
                or isinstance(state.transformer_context_length, bool)
                or state.frame_position <= 0
                or state.transformer_context_length <= 0
                or state.transformer_context_length > state.frame_position
                or state.transformer_context_length
                != min(retained_context, state.frame_position)
            ):
                raise ValueError("incremental jobs require coherent warm Codec states")
            try:
                signature = cls._mapping_signature(state)
            except ValueError as error:
                raise ValueError(
                    "incremental jobs require coherent warm Codec states"
                ) from error
            if expected_signature is None:
                expected_signature = signature
            elif signature != expected_signature:
                raise ValueError("incremental jobs require coherent warm Codec states")

    @staticmethod
    def _pcm16(wav: torch.Tensor) -> torch.Tensor:
        return (wav.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)

    def _require_frames(
        self,
        frames: torch.Tensor,
        *,
        states: list[IncrementalCodecState]
        | tuple[IncrementalCodecState, ...]
        | None = None,
    ) -> None:
        if frames.ndim != 3:
            raise ValueError("Codec frames must have shape (rows, frames, codebooks)")
        if frames.shape[0] < 1:
            raise ValueError("Codec jobs require at least one row")
        if frames.shape[2] != self.num_code_groups:
            raise ValueError(
                f"Codec frame codebook width must be {self.num_code_groups}, "
                f"got {frames.shape[2]}"
            )
        if frames.dtype != torch.long:
            raise TypeError("Codec frames must use torch.long")
        if states is not None:
            for state in states:
                for name in _STATE_MAPPINGS:
                    if any(
                        value.device != frames.device
                        for value in getattr(state, name).values()
                    ):
                        raise ValueError(
                            "Codec frames and state tensors must share one device"
                        )

    @torch.inference_mode()
    def whole_sequence_decode(self, codec_window: torch.Tensor) -> CodecResult:
        self._require_frames(codec_window)
        if codec_window.shape[1] == 0:
            raise ValueError(
                "whole-sequence Codec window must have shape (rows, frames, codebooks)"
            )
        wav = self.model.decoder(codec_window.transpose(1, 2)).squeeze(1)
        expected = codec_window.shape[1] * self.samples_per_frame
        if wav.ndim != 2 or wav.shape[0] != codec_window.shape[0]:
            raise RuntimeError(
                f"whole-sequence Codec produced an invalid waveform shape {tuple(wav.shape)}"
            )
        if wav.shape[1] < expected:
            raise RuntimeError(
                f"whole-sequence Codec produced {wav.shape[1]} samples, expected {expected}"
            )
        return CodecResult(
            pcm=self._pcm16(wav[:, :expected]),
            states=None,
            terminal=False,
        )

    def _incremental(
        self,
        frames: torch.Tensor,
        states: list[IncrementalCodecState] | tuple[IncrementalCodecState, ...],
        *,
        terminal: bool,
    ) -> CodecResult:
        self._require_frames(frames, states=states)
        if frames.shape[0] != len(states):
            raise ValueError(
                "incremental Codec frames and request states must have equal rows"
            )
        owned = tuple(self.clone_state(state) for state in states)
        if frames.shape[1] == 0:
            if not terminal:
                raise ValueError("only a terminal Codec job may contain zero frames")
            pcm = frames.new_empty((frames.shape[0], 0), dtype=torch.int16)
            return CodecResult(pcm=pcm, states=owned, terminal=True)
        wav = self.incremental_decoder(
            frames.transpose(1, 2),
            list(owned),
        ).squeeze(1)
        expected = frames.shape[1] * self.samples_per_frame
        if wav.shape[1] != expected:
            raise RuntimeError(
                f"incremental Codec produced {wav.shape[1]} samples, expected {expected}"
            )
        return CodecResult(
            pcm=self._pcm16(wav),
            states=owned,
            terminal=terminal,
        )

    @torch.inference_mode()
    def state_bootstrap(
        self,
        frames: torch.Tensor,
        states: list[IncrementalCodecState] | tuple[IncrementalCodecState, ...],
    ) -> CodecResult:
        self._require_cold_states(states)
        return self._incremental(frames, states, terminal=False)

    @torch.inference_mode()
    def warm_incremental(
        self,
        frames: torch.Tensor,
        states: list[IncrementalCodecState] | tuple[IncrementalCodecState, ...],
    ) -> CodecResult:
        self._require_warm_states(states, retained_context=self.retained_context)
        return self._incremental(frames, states, terminal=False)

    @torch.inference_mode()
    def terminal(
        self,
        frames: torch.Tensor,
        states: list[IncrementalCodecState] | tuple[IncrementalCodecState, ...],
    ) -> CodecResult:
        self._require_warm_states(states, retained_context=self.retained_context)
        return self._incremental(frames, states, terminal=True)

    @property
    def captured_cuda_graph_instances(self) -> int:
        return len(self._slots)

    def _template(self) -> IncrementalCodecState:
        """Return a representative post-bootstrap state to size warm buffers.

        Warm state tensors are fixed-size -- the valid retained-context length
        travels separately in ``context_lengths`` -- so one template serves
        every warm shape and every request phase. It is only read for its
        schema and values; ``_state_inputs`` copies out of it, so the cached
        instance is never handed to a captured CUDA Graph.
        """

        if self._warm_template is None:
            state = self.new_state()
            frames = torch.zeros(
                (1, WARM_TEMPLATE_FRAMES, self.num_code_groups),
                dtype=torch.long,
                device=self.device,
            )
            self.incremental_decoder(
                frames.transpose(1, 2),
                [state],
                position_ids=torch.arange(WARM_TEMPLATE_FRAMES, device=self.device).unsqueeze(0),
                context_lengths=torch.zeros(1, dtype=torch.long, device=self.device),
            )
            self._warm_template = state
        return self._warm_template

    @staticmethod
    def _state_inputs(template: IncrementalCodecState, rows: int) -> tuple[_StateBuffer, ...]:
        values: list[_StateBuffer] = []
        for mapping_name in _STATE_MAPPINGS:
            mapping = getattr(template, mapping_name)
            for key in sorted(mapping, key=str):
                tensor = mapping[key]
                values.append(
                    _StateBuffer(
                        mapping_name,
                        key,
                        tensor.unsqueeze(0).expand(rows, *tensor.shape).clone(),
                    )
                )
        return tuple(values)

    def _capture_state(self, key: CodecCaptureKey) -> _CodecCaptureState:
        template = self._template() if key.mode is CodecExecutionMode.WARM else self.new_state()
        rows = key.capture_batch_size
        context = template.transformer_context_length if key.mode is CodecExecutionMode.WARM else 0
        base_position_ids = (
            torch.arange(key.model_frames, device=self.device)
            .unsqueeze(0)
            .expand(rows, -1)
            .clone()
        )
        metadata = torch.zeros((2, rows), dtype=torch.long, device=self.device)
        metadata[0].fill_(template.frame_position)
        metadata[1].fill_(context)
        host_metadata = torch.zeros(
            (2, rows),
            dtype=torch.long,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        host_metadata[0].fill_(template.frame_position)
        host_metadata[1].fill_(context)
        return _CodecCaptureState(
            inputs=self._state_inputs(template, rows),
            base_position_ids=base_position_ids,
            metadata=metadata,
            host_metadata=host_metadata,
        )

    def _decode_incremental(self, frames: torch.Tensor, capture_state: _CodecCaptureState) -> torch.Tensor:
        states = [self.new_state() for _ in range(frames.shape[0])]
        for entry in capture_state.inputs:
            for row, state in enumerate(states):
                getattr(state, entry.mapping)[entry.key] = entry.values[row]
        wav = self.incremental_decoder(
            frames.transpose(1, 2),
            states,
            position_ids=(
                capture_state.base_position_ids
                + capture_state.metadata[0].unsqueeze(1)
            ),
            context_lengths=capture_state.metadata[1],
        ).squeeze(1)
        capture_state.outputs = tuple(states)
        return self._pcm16(wav)

    def capture(self, key: CodecCaptureKey) -> None:
        if not isinstance(key, CodecCaptureKey):
            raise TypeError("Codec executor received the wrong capture key")
        if key in self._slots:
            return
        frames = torch.zeros(
            (key.capture_batch_size, key.model_frames, self.num_code_groups),
            dtype=torch.long,
            device=self.device,
        )
        if key.mode is CodecExecutionMode.WHOLE_SEQUENCE:
            state = None

            def operation() -> torch.Tensor:
                return self.whole_sequence_decode(frames).pcm

        else:
            state = self._capture_state(key)

            def operation() -> torch.Tensor:
                assert state is not None
                return self._decode_incremental(frames, state)

        self._slots[key] = _CodecSlot(frames, state, self.driver.capture(operation), SlotLeaseState())

    @staticmethod
    def _capture_cold(state: IncrementalCodecState) -> bool:
        return (
            type(state.frame_position) is int
            and type(state.transformer_context_length) is int
            and state.frame_position == 0
            and state.transformer_context_length == 0
            and all(
                isinstance(getattr(state, name), dict)
                and not getattr(state, name)
                for name in _STATE_MAPPINGS
            )
        )

    @staticmethod
    def _capture_warm(state: IncrementalCodecState) -> bool:
        return (
            type(state.frame_position) is int
            and type(state.transformer_context_length) is int
            and state.frame_position > 0
            and 0 < state.transformer_context_length <= state.frame_position
            and all(
                isinstance(getattr(state, name), dict)
                and getattr(state, name)
                for name in _STATE_MAPPINGS
            )
            and state.transformer_keys.keys() == state.transformer_values.keys()
        )

    def _stage_state(  # noqa: PLR0912 - fail-closed schema validation plus grouped staging
        self,
        key: CodecCaptureKey,
        capture: _CodecCaptureState,
        states: tuple[IncrementalCodecState, ...],
    ) -> None:
        capture.host_metadata.zero_()
        logical_rows = len(states)
        if logical_rows < key.capture_batch_size:
            padded = tuple(
                entry.values[logical_rows:]
                for entry in capture.inputs
                if entry.values.shape[0] > logical_rows
            )
            if padded:
                torch._foreach_zero_(padded)
        copy_groups: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        for row, state in enumerate(states):
            if key.mode is CodecExecutionMode.COLD:
                if not self._capture_cold(state):
                    raise ValueError(
                        "cold Codec capture requires coherent empty request state"
                    )
                continue
            if not self._capture_warm(state):
                raise ValueError("warm Codec capture requires coherent request state")
            for mapping_name in _STATE_MAPPINGS:
                expected_keys = {
                    entry.key for entry in capture.inputs if entry.mapping == mapping_name
                }
                actual_keys = set(getattr(state, mapping_name))
                if actual_keys != expected_keys:
                    raise ValueError(
                        "warm Codec request state mapping keys do not match the captured schema"
                    )
            capture.host_metadata[0, row] = state.frame_position
            capture.host_metadata[1, row] = state.transformer_context_length
            for entry in capture.inputs:
                value = getattr(state, entry.mapping).get(entry.key)
                if value is None or value.shape != entry.values.shape[1:]:
                    raise ValueError("warm Codec request state does not match the captured state shape")
                if value.dtype != entry.values.dtype:
                    raise ValueError("warm Codec request state dtype does not match the captured schema")
                if value.device != entry.values.device:
                    raise ValueError("warm Codec request state device does not match the captured schema")
                destinations, sources = copy_groups.setdefault(
                    (value.device, value.dtype),
                    ([], []),
                )
                destinations.append(entry.values[row])
                sources.append(value)
        capture.metadata.copy_(
            capture.host_metadata,
            non_blocking=self.device.type == "cuda",
        )
        for destinations, sources in copy_groups.values():
            torch._foreach_copy_(destinations, sources)

    def _clone_states(
        self,
        key: CodecCaptureKey,
        capture: _CodecCaptureState,
        source: tuple[IncrementalCodecState, ...],
    ) -> tuple[IncrementalCodecState, ...]:
        results: list[IncrementalCodecState] = []
        allocation_groups: dict[
            tuple[torch.device, torch.dtype],
            list[tuple[dict[int | str, torch.Tensor], int | str, torch.Tensor]],
        ] = {}
        for row, previous in enumerate(source):
            captured = capture.outputs[row]
            result = self.new_state()
            result.frame_position = previous.frame_position + key.model_frames
            result.transformer_context_length = min(
                self.retained_context,
                previous.transformer_context_length + key.model_frames,
            )
            for name in _STATE_MAPPINGS:
                owned: dict[int | str, torch.Tensor] = {}
                for mapping_key, value in getattr(captured, name).items():
                    allocation_groups.setdefault((value.device, value.dtype), []).append(
                        (owned, mapping_key, value.detach())
                    )
                setattr(result, name, owned)
            results.append(result)

        # A Codec successor contains dozens of small request-local tensors.
        # Allocate one backing store per device/dtype and expose shaped views;
        # this preserves owned completion output without paying one allocator
        # submission per tensor. The grouped copy remains completion-ordered.
        for (device, dtype), entries in allocation_groups.items():
            backing = torch.empty(
                sum(value.numel() for _, _, value in entries),
                device=device,
                dtype=dtype,
            )
            destinations: list[torch.Tensor] = []
            sources: list[torch.Tensor] = []
            offset = 0
            for owned, mapping_key, value in entries:
                destination = backing.narrow(0, offset, value.numel()).view(value.shape)
                owned[mapping_key] = destination
                destinations.append(destination)
                sources.append(value)
                offset += value.numel()
            torch._foreach_copy_(destinations, sources)
        return tuple(results)

    def replay(self, key: CodecCaptureKey, values: CodecRowsExecutionInput) -> CodecResult:
        submission = self.submission_fence.reserve()
        try:
            return self._replay_owned(key, values)
        finally:
            self.submission_fence.release(submission)

    def _replay_owned(  # noqa: PLR0912,PLR0915 - fixed Codec lifecycle validation
        self,
        key: CodecCaptureKey,
        values: CodecRowsExecutionInput,
    ) -> CodecResult:
        if not isinstance(key, CodecCaptureKey):
            raise TypeError("Codec executor received the wrong capture key")
        if not isinstance(values, CodecRowsExecutionInput):
            raise TypeError("Codec replay requires a typed Codec input")
        slot = self._slots.get(key)
        if slot is None:
            raise RuntimeError("Codec CUDA Graph has not been captured")
        logical_rows = len(values.rows)
        frame_counts = tuple(len(row.frames) for row in values.rows)
        logical_frames = max(frame_counts, default=0)
        terminal_rows = tuple(
            values.terminal if row.terminal is None else row.terminal
            for row in values.rows
        )
        if len(set(terminal_rows)) != 1:
            raise ValueError("Codec rows must agree on terminal lifecycle")
        terminal = terminal_rows[0]
        if len(set(frame_counts)) != 1 and not (
            key.mode is CodecExecutionMode.WARM and terminal
        ):
            raise ValueError("Codec row frame counts must be exact-compatible")
        if any(
            frame.numel() != self.num_code_groups
            for row in values.rows
            for frame in row.frames
        ):
            raise ValueError("Codec row frames have an invalid codebook shape")
        if any(
            frame.device != self.device
            for row in values.rows
            for frame in row.frames
        ):
            raise ValueError("Codec row frames must be on the execution device")
        states = tuple(row.state for row in values.rows)
        if logical_frames > key.model_frames or logical_rows > key.capture_batch_size:
            raise ValueError("Codec input exceeds its captured shape")
        if any(count != key.model_frames for count in frame_counts) and not terminal:
            raise ValueError(
                "Codec nonterminal replay requires exact captured frames; only terminal padding is valid"
            )
        pcm_start_frames = tuple(
            values.pcm_start_frame
            if row.pcm_start_frame is None
            else row.pcm_start_frame
            for row in values.rows
        )
        visible_frames = tuple(
            values.visible_frames
            if row.visible_frames is None
            else row.visible_frames
            for row in values.rows
        )
        if any(
            start + visible > count
            for start, visible, count in zip(
                pcm_start_frames,
                visible_frames,
                frame_counts,
                strict=True,
            )
        ):
            raise ValueError("Codec PCM frame window exceeds the logical input")
        lease = slot.lease_state.reserve()
        try:
            slot.frames.zero_()
            for row_index, row in enumerate(values.rows):
                row_frames = len(row.frames)
                if row_frames == 1:
                    slot.frames[row_index, 0].copy_(row.frames[0].reshape(-1))
                elif logical_rows == 1:
                    torch._foreach_copy_(
                        tuple(slot.frames[row_index, frame] for frame in range(row_frames)),
                        tuple(frame.reshape(-1) for frame in row.frames),
                    )
                else:
                    torch.stack(
                        tuple(frame.reshape(-1) for frame in row.frames),
                        out=slot.frames[row_index, :row_frames],
                    )
            if key.mode is CodecExecutionMode.WHOLE_SEQUENCE:
                if (states is not None and any(state is not None for state in states)) or slot.state is not None:
                    raise ValueError("whole-sequence Codec replay does not accept incremental state")
                typed_states = None
            else:
                if (
                    states is None
                    or len(states) != logical_rows
                    or any(state is None for state in states)
                    or slot.state is None
                ):
                    raise ValueError("incremental Codec replay requires one state per row")
                typed_states = tuple(state for state in states if state is not None)
                self._stage_state(key, slot.state, typed_states)
            captured = slot.call.replay()
            if not isinstance(captured, torch.Tensor):
                raise RuntimeError("Codec CUDA Graph returned an invalid result")
            sample_starts = tuple(
                start * self.samples_per_frame for start in pcm_start_frames
            )
            sample_lengths = tuple(
                frames * self.samples_per_frame for frames in visible_frames
            )
            maximum_length = max(sample_lengths, default=0)
            if len(set(sample_starts)) == 1:
                sample_start = sample_starts[0]
                pcm = captured[
                    :logical_rows,
                    sample_start : sample_start + maximum_length,
                ].clone()
            else:
                pcm = captured.new_zeros((logical_rows, maximum_length))
                copies = tuple(
                    (pcm[row, :length], captured[row, start : start + length])
                    for row, (start, length) in enumerate(
                        zip(sample_starts, sample_lengths, strict=True)
                    )
                    if length > 0
                )
                if copies:
                    torch._foreach_copy_(
                        tuple(destination for destination, _source in copies),
                        tuple(source for _destination, source in copies),
                    )
            output_states = (
                None
                if slot.state is None
                else self._clone_states(key, slot.state, typed_states or ())
            )
            return CodecResult(
                pcm=pcm,
                states=output_states,
                terminal=terminal,
                pcm_lengths=(
                    None
                    if len(set(sample_lengths)) == 1
                    else sample_lengths
                ),
            )
        finally:
            slot.lease_state.release(lease)


__all__ = ["CodecExecutor"]

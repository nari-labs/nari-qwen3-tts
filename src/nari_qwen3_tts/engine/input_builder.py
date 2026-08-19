"""Engine-owned construction of typed execution inputs from request state."""

from __future__ import annotations

import torch

from nari_qwen3_tts.contract.codec_state import IncrementalCodecState
from nari_qwen3_tts.contract.model import SynthesisModelSpec
from nari_qwen3_tts.contract.request import AdmittedRequest
from nari_qwen3_tts.contract.rng import logical_rng_offset
from nari_qwen3_tts.contract.stage import CodecExecutionMode, SynthesisStage
from nari_qwen3_tts.contract.work import (
    CodecBatchCompatibility,
    CodePredictorBatchCompatibility,
    ScheduleDecision,
    StageExecutionBatch,
)
from nari_qwen3_tts.engine.state import RequestState, RequestStateStore
from nari_qwen3_tts.executor.rows import (
    CodecExecutionRow,
    CodecMetadataExecutionRow,
    CodecRowsExecutionInput,
    CodePredictorExecutionRow,
    TalkerDecodeExecutionRow,
    TalkerPrefillExecutionRow,
    TalkerSamplingExecutionRow,
)
from nari_qwen3_tts.executor.submission import StageExecutionInputs


class InputBuilder:
    """The only dispatch from a planned stage row to executor input values."""

    def __init__(self, *, model_spec: SynthesisModelSpec) -> None:
        self.model_spec = model_spec

    @staticmethod
    def _request_input(state: RequestState) -> AdmittedRequest:
        value = state.input
        if not isinstance(value, AdmittedRequest):
            raise RuntimeError("request lacks its immutable admitted input")
        return value

    def _sampling(self, state: RequestState) -> TalkerSamplingExecutionRow:
        request = self._request_input(state).request
        return TalkerSamplingExecutionRow(
            temperature=request.temperature if request.do_sample else 0.0,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.random_seed,
            offset=state.generation.next_sampling_offset,
            seen_token_mask=(
                state.generation.seen_token_mask
                if isinstance(state.generation.seen_token_mask, torch.Tensor)
                else None
            ),
        )

    def _suppress_decode_eos(self, state: RequestState) -> bool:
        if self._request_input(state).request.ignore_eos:
            return True
        generation_step = state.generation.generation_step
        if generation_step < 2:
            return True
        continuation = self._request_input(state).talker_input.continuation
        if continuation.terminal_token_id is None:
            return False
        return not continuation.input_finished or generation_step < continuation.token_count - 1

    def _requires_host_finalize(self, states: tuple[RequestState, ...]) -> bool:
        return any(
            not self._suppress_decode_eos(state)
            and not (request := self._request_input(state).request).ignore_eos
            and state.generation.generation_step + 1 < request.effective_max_output_tokens
            for state in states
        )

    def _prefill(
        self,
        batch: StageExecutionBatch,
        states: tuple[RequestState, ...],
    ) -> StageExecutionInputs:
        rows = tuple(
            TalkerPrefillExecutionRow(
                text_token_ids=(request_input := self._request_input(state)).talker_input.text_token_ids,
                codec_token_ids=request_input.talker_input.codec_token_ids,
                codec_token_mask=request_input.talker_input.codec_token_mask,
                suppress_eos=True,
                sampling=self._sampling(state),
            )
            for state in states
        )
        return StageExecutionInputs(batch.batch_id, rows)

    def _decode(
        self,
        batch: StageExecutionBatch,
        states: tuple[RequestState, ...],
    ) -> StageExecutionInputs:
        rows: list[TalkerDecodeExecutionRow] = []
        for state in states:
            step_input = state.generation.step_input
            if not isinstance(step_input, torch.Tensor):
                raise RuntimeError("Talker decode requires a committed Code Predictor step input")
            continuation = self._request_input(state).talker_input.continuation
            rows.append(
                TalkerDecodeExecutionRow(
                    talker_step_embed=step_input,
                    text_token_id=continuation.token_at(state.generation.generation_step),
                    suppress_eos=self._suppress_decode_eos(state),
                    sampling=self._sampling(state),
                )
            )
        return StageExecutionInputs(
            batch.batch_id,
            tuple(rows),
            requires_host_finalize=self._requires_host_finalize(states),
            reuse_attention_plan=all(
                not self._request_input(state).talker_input.continuation.non_streaming_mode
                for state in states
            ),
        )

    def _code_predictor(
        self,
        batch: StageExecutionBatch,
        states: tuple[RequestState, ...],
    ) -> StageExecutionInputs:
        compatibility = batch.compatibility
        if not isinstance(compatibility, CodePredictorBatchCompatibility):
            raise RuntimeError("Code Predictor plan requires typed compatibility")
        rows: list[CodePredictorExecutionRow] = []
        for state in states:
            hidden = state.generation.hidden
            token = state.generation.token
            if not isinstance(hidden, torch.Tensor) or not isinstance(token, torch.Tensor):
                raise RuntimeError("Code Predictor requires committed Talker token and hidden state")
            request = self._request_input(state).request
            rows.append(
                CodePredictorExecutionRow(
                    layer0_token=token,
                    past_hidden=hidden,
                    temperature=(
                        request.subtalker_temperature if request.subtalker_dosample else 0.0
                    ),
                    top_k=request.subtalker_top_k,
                    top_p=request.subtalker_top_p,
                    seed=request.random_seed,
                    offsets=tuple(
                        logical_rng_offset(state.generation.frame_index, codebook)
                        for codebook in range(1, self.model_spec.num_codebooks)
                    ),
                )
            )
        return StageExecutionInputs(batch.batch_id, tuple(rows))

    def _codec(
        self,
        batch: StageExecutionBatch,
        states: tuple[RequestState, ...],
    ) -> StageExecutionInputs:
        if batch.capture is None:
            return StageExecutionInputs(
                batch.batch_id,
                tuple(CodecMetadataExecutionRow() for _ in states),
            )
        compatibility = batch.compatibility
        if not isinstance(compatibility, CodecBatchCompatibility):
            raise RuntimeError("Codec plan requires typed compatibility")
        rows: list[CodecExecutionRow] = []
        real_rows = tuple(row for row in batch.rows if not row.padding)
        for manifest, state in zip(real_rows, states, strict=True):
            row_compatibility = manifest.compatibility
            if not isinstance(row_compatibility, CodecBatchCompatibility):
                raise RuntimeError("Codec row requires typed compatibility")
            if row_compatibility.mode in {
                CodecExecutionMode.WHOLE_SEQUENCE,
                CodecExecutionMode.COLD,
            }:
                frames = state.codec.history_frames[: row_compatibility.input_frames]
            else:
                frames = state.codec.buffered_frames[: row_compatibility.producer_frames]
            decoder_state = None
            if row_compatibility.mode is CodecExecutionMode.COLD:
                decoder_state = IncrementalCodecState()
            elif row_compatibility.mode is CodecExecutionMode.WARM:
                decoder_state = state.codec.decoder_state
                if decoder_state is None:
                    raise RuntimeError("warm Codec work requires committed decoder state")
            rows.append(
                CodecExecutionRow(
                    frames=frames,
                    state=decoder_state,
                    visible_frames=row_compatibility.visible_frames,
                    pcm_start_frame=row_compatibility.pcm_start_frame,
                    terminal=row_compatibility.terminal,
                )
            )
        call = CodecRowsExecutionInput(
            rows=tuple(rows),
            visible_frames=compatibility.visible_frames,
            pcm_start_frame=compatibility.pcm_start_frame,
            terminal=compatibility.terminal,
        )
        return StageExecutionInputs(batch.batch_id, call.rows)

    def _build_one(
        self,
        batch: StageExecutionBatch,
        states: tuple[RequestState, ...],
    ) -> StageExecutionInputs:
        if batch.stage is SynthesisStage.TALKER_PREFILL:
            return self._prefill(batch, states)
        if batch.stage is SynthesisStage.TALKER_DECODE:
            return self._decode(batch, states)
        if batch.stage is SynthesisStage.CODE_PREDICTOR:
            return self._code_predictor(batch, states)
        return self._codec(batch, states)

    def build(
        self,
        decision: ScheduleDecision,
        store: RequestStateStore,
    ) -> tuple[StageExecutionInputs, ...]:
        return tuple(
            self.build_batch(
                batch,
                tuple(store.request(request_id) for request_id in batch.request_ids),
            )
            for batch in decision.batches
        )

    def build_batch(
        self,
        batch: StageExecutionBatch,
        states: tuple[RequestState, ...],
    ) -> StageExecutionInputs:
        return self._build_one(batch, states)

    def suppress_decode_eos(self, state: RequestState) -> bool:
        return self._suppress_decode_eos(state)


__all__ = ["InputBuilder"]

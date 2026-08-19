"""Typed dispatch and lifecycle accounting for fixed CUDA executors."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nari_qwen3_tts.contract.health import EngineHealth, StageStats
from nari_qwen3_tts.contract.stage import (
    CodecCaptureKey,
    CodePredictorCaptureKey,
    CudaGraphKey,
    SynthesisStage,
    TalkerDecodeCaptureKey,
    TalkerPrefillCaptureKey,
)
from nari_qwen3_tts.contract.work import (
    CodecBatchCompatibility,
    CodePredictorBatchCompatibility,
    ScheduleDecision,
    StageExecutionBatch,
    TalkerPrefillBatchCompatibility,
)
from nari_qwen3_tts.executor.code_predictor import CodePredictorExecutor
from nari_qwen3_tts.executor.codec import CodecExecutor
from nari_qwen3_tts.executor.cuda_graph import CudaSubmissionFence
from nari_qwen3_tts.executor.health import (
    CaptureStartupError,
    UncapturedExecutionError,
)
from nari_qwen3_tts.executor.optimizations import (
    ExecutionOptimizationReport,
)
from nari_qwen3_tts.executor.rows import (
    CodecExecutionRow,
    CodecMetadataExecutionRow,
    CodecRowsExecutionInput,
    CodePredictorExecutionRow,
    CodePredictorRowsExecutionInput,
    TalkerDecodeExecutionRow,
    TalkerDecodeRowsExecutionInput,
    TalkerExecutionResult,
    TalkerPrefillExecutionRow,
    TalkerPrefillRowsExecutionInput,
)
from nari_qwen3_tts.executor.submission import (
    StageExecutionInputs,
    StageExecutionSubmission,
)
from nari_qwen3_tts.executor.talker import TalkerExecutor
from nari_qwen3_tts.executor.types import CodecResult, CodePredictorResult
from nari_qwen3_tts.profile import ResolvedProfile


@dataclass(slots=True)
class Executor:
    config: ResolvedProfile
    required_keys: frozenset[CudaGraphKey] | None
    talker: TalkerExecutor
    code_predictor: CodePredictorExecutor
    codec: CodecExecutor
    optimizations: ExecutionOptimizationReport
    completion_event_factory: object | None = None
    _captured: set[CudaGraphKey] = field(default_factory=set, init=False, repr=False)
    _capture_failures: int = field(default=0, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _startup_complete: bool = field(default=False, init=False, repr=False)
    _submitted: dict[SynthesisStage, int] = field(
        default_factory=lambda: dict.fromkeys(SynthesisStage, 0),
        init=False,
        repr=False,
    )
    _replayed: dict[SynthesisStage, int] = field(
        default_factory=lambda: dict.fromkeys(SynthesisStage, 0),
        init=False,
        repr=False,
    )
    _failed: dict[SynthesisStage, int] = field(
        default_factory=lambda: dict.fromkeys(SynthesisStage, 0),
        init=False,
        repr=False,
    )
    _metadata_actions: int = field(default=0, init=False, repr=False)
    _required_key_count: int = field(default=0, init=False, repr=False)
    _required_instance_count: int = field(default=0, init=False, repr=False)
    _capture_executors: tuple[object, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        unique: dict[int, object] = {}
        for executor in (self.talker, self.code_predictor, self.codec):
            unique[id(executor)] = executor
        self._capture_executors = tuple(unique.values())
        if self.required_keys is None:
            return

        talker_keys = 0
        code_predictor_keys = 0
        codec_keys = 0
        for key in self.required_keys:
            if isinstance(key, (TalkerPrefillCaptureKey, TalkerDecodeCaptureKey)):
                talker_keys += 1
            elif isinstance(key, CodePredictorCaptureKey):
                code_predictor_keys += 1
            elif isinstance(key, CodecCaptureKey):
                codec_keys += 1
        self._required_key_count = len(self.required_keys)
        self._required_instance_count = (
            talker_keys * int(getattr(self.talker, "capture_slots", 1))
            + code_predictor_keys
            * int(getattr(self.code_predictor, "capture_instances_per_key", 1))
            + codec_keys
        )

    def add_request(self, request_id: str) -> None:
        self.talker.add_request(request_id)

    def remove_request(self, request_id: str) -> None:
        self.talker.remove_request(request_id)

    def talker_prefill_rows(
        self,
        key: TalkerPrefillCaptureKey,
        values: TalkerPrefillRowsExecutionInput,
    ) -> TalkerExecutionResult:
        output = self._replay_stage(
            SynthesisStage.TALKER_PREFILL,
            TalkerPrefillCaptureKey,
            self.talker,
            key,
            values,
        )
        if not isinstance(output, TalkerExecutionResult):
            raise RuntimeError("Talker prefill returned an invalid typed result")
        return output

    def talker_decode_rows(
        self,
        key: TalkerDecodeCaptureKey,
        values: TalkerDecodeRowsExecutionInput,
    ) -> TalkerExecutionResult:
        output = self._replay_stage(
            SynthesisStage.TALKER_DECODE,
            TalkerDecodeCaptureKey,
            self.talker,
            key,
            values,
        )
        if not isinstance(output, TalkerExecutionResult):
            raise RuntimeError("Talker decode returned an invalid typed result")
        return output

    def code_predictor_rows(
        self,
        key: CodePredictorCaptureKey,
        values: CodePredictorRowsExecutionInput,
    ) -> CodePredictorResult:
        output = self._replay_stage(
            SynthesisStage.CODE_PREDICTOR,
            CodePredictorCaptureKey,
            self.code_predictor,
            key,
            values,
        )
        if not isinstance(output, CodePredictorResult):
            raise RuntimeError("Code Predictor returned an invalid typed result")
        return output

    def codec_rows(
        self,
        key: CodecCaptureKey,
        values: CodecRowsExecutionInput,
    ) -> CodecResult:
        output = self._replay_stage(
            SynthesisStage.CODEC,
            CodecCaptureKey,
            self.codec,
            key,
            values,
        )
        if not isinstance(output, CodecResult):
            raise RuntimeError("Codec returned an invalid typed result")
        return output

    def empty_terminal(self, *, rows: int) -> None:
        if rows <= 0:
            raise ValueError("empty terminal action requires request rows")
        self._metadata_actions += 1

    @staticmethod
    def _ordered_prefill(keys: tuple[TalkerPrefillCaptureKey, ...]) -> list[TalkerPrefillCaptureKey]:
        return sorted(
            keys,
            key=lambda key: (
                key.capture_sequence_length is None,
                key.capture_batch_size,
                key.token_capacity,
            ),
        )

    def capture_all(self) -> None:
        if self._started:
            raise CaptureStartupError("CUDA capture startup may run only once")
        self._started = True
        if self.required_keys is None:
            raise CaptureStartupError("production capture startup requires declared CUDA graph keys")
        talker_prefill = tuple(
            key for key in self.required_keys if isinstance(key, TalkerPrefillCaptureKey)
        )
        talker_decode = tuple(
            key for key in self.required_keys if isinstance(key, TalkerDecodeCaptureKey)
        )
        code_predictor = tuple(
            key for key in self.required_keys if isinstance(key, CodePredictorCaptureKey)
        )
        codec = tuple(key for key in self.required_keys if isinstance(key, CodecCaptureKey))
        ordered: tuple[tuple[object, list[CudaGraphKey]], ...] = (
            (
                self.talker,
                sorted(talker_decode, key=lambda key: -key.capture_batch_size),
            ),
            (self.talker, self._ordered_prefill(talker_prefill)),
            (
                self.codec,
                sorted(
                    codec,
                    key=lambda key: (
                        key.mode.value,
                        -key.capture_batch_size,
                        -key.model_frames,
                    ),
                ),
            ),
            (
                self.code_predictor,
                sorted(
                    code_predictor,
                    key=lambda key: -key.capture_batch_size,
                ),
            ),
        )
        for executor, keys in ordered:
            for key in keys:
                try:
                    executor.capture(key)
                except Exception as error:
                    self._capture_failures += 1
                    raise CaptureStartupError(f"CUDA capture failed for {key!r}") from error
                self._captured.add(key)
        self._startup_complete = True

    def _replay_stage(
        self,
        stage: SynthesisStage,
        expected_key: type,
        executor: object,
        key: object,
        values: object,
    ) -> object:
        self._submitted[stage] += 1
        if not isinstance(key, expected_key):
            self._failed[stage] += 1
            label = {
                SynthesisStage.TALKER_PREFILL: "Talker prefill",
                SynthesisStage.TALKER_DECODE: "Talker decode",
                SynthesisStage.CODE_PREDICTOR: "Code Predictor",
                SynthesisStage.CODEC: "Codec",
            }[stage]
            raise TypeError(f"{label} replay requires its typed capture key")
        if self.required_keys is not None:
            if not self._startup_complete:
                self._failed[stage] += 1
                raise UncapturedExecutionError("CUDA Graph startup did not complete")
            if key not in self._captured:
                self._failed[stage] += 1
                raise UncapturedExecutionError(
                    f"non-empty {stage.value} work has no captured CUDA Graph for {key!r}"
                )
        try:
            result = executor.replay(key, values)
        except Exception:
            self._failed[stage] += 1
            raise
        self._replayed[stage] += 1
        return result

    def health(self) -> EngineHealth:
        if self.required_keys is None:
            raise RuntimeError("execution health requires declared CUDA graph keys")
        captured_instances = sum(
            int(
                getattr(
                    executor,
                    "captured_cuda_graph_instances",
                    len(getattr(executor, "captured", ())),
                )
            )
            for executor in self._capture_executors
        )
        return EngineHealth(
            required_keys=self._required_key_count,
            captured_keys=len(self._captured),
            required_cuda_graph_instances=self._required_instance_count,
            captured_cuda_graph_instances=captured_instances,
            capture_failures=self._capture_failures,
            eager_fallbacks=0,
            stages=tuple(
                StageStats(
                    stage,
                    self._submitted[stage],
                    self._replayed[stage],
                    self._failed[stage],
                )
                for stage in SynthesisStage
            ),
            metadata_actions=self._metadata_actions,
        )

    @staticmethod
    def _capture_capacity(batch: StageExecutionBatch) -> int:
        if batch.capture is None:
            return batch.logical_rows
        return int(batch.capture.key.capture_batch_size)

    def _require_captured_batch(self, batch: StageExecutionBatch) -> None:
        if self._capture_capacity(batch) != len(batch.rows):
            raise ValueError("CUDA graph capture capacity does not match physical rows")
        if batch.capture is None:
            return
        key = batch.capture.key
        if self.required_keys is not None and key not in self.required_keys:
            raise ValueError("stage execution capture key is outside the resolved catalog")
        if self.required_keys is not None and key not in self._captured:
            raise RuntimeError("stage execution capture key was not captured")

    @staticmethod
    def _preflight_prefill(
        batch: StageExecutionBatch,
        values: StageExecutionInputs,
    ) -> None:
        assert isinstance(batch.compatibility, TalkerPrefillBatchCompatibility)
        assert batch.capture is not None
        key = batch.capture.key
        lengths = tuple(row.text_token_ids.numel() for row in values.rows)
        declared = tuple(
            row.compatibility.sequence_length
            for row in batch.rows[: batch.logical_rows]
        )
        if lengths != declared:
            raise ValueError("Talker prefill input lengths do not match batch rows")
        if sum(lengths) > key.token_capacity:
            raise ValueError("Talker prefill inputs exceed capture token capacity")

    @staticmethod
    def _preflight_codec(
        batch: StageExecutionBatch,
        values: StageExecutionInputs,
    ) -> None:
        if not isinstance(batch.compatibility, CodecBatchCompatibility):
            raise TypeError("Codec batch compatibility is invalid")
        if batch.capture is None:
            if any(type(row) is not CodecMetadataExecutionRow for row in values.rows):
                raise ValueError("empty Codec metadata requires metadata-only rows")
            return
        if any(type(row) is not CodecExecutionRow for row in values.rows):
            raise ValueError("captured Codec execution requires tensor rows")
        key = batch.capture.key
        if (
            key.mode is not batch.compatibility.mode
            or key.model_frames != batch.compatibility.model_frames
        ):
            raise ValueError("Codec capture key does not match lifecycle compatibility")
        if any(len(row.frames) > key.model_frames for row in values.rows):
            raise ValueError("Codec inputs exceed capture frame capacity")

    def _preflight_batch(
        self,
        batch: StageExecutionBatch,
        values: StageExecutionInputs,
    ) -> None:
        if values.batch_id != batch.batch_id:
            raise ValueError("stage execution input batch ID does not match")
        if len(values.rows) != batch.logical_rows:
            raise ValueError("stage execution input count does not match logical rows")
        self._require_captured_batch(batch)

        expected: type | tuple[type, ...] = {
            SynthesisStage.TALKER_PREFILL: TalkerPrefillExecutionRow,
            SynthesisStage.TALKER_DECODE: TalkerDecodeExecutionRow,
            SynthesisStage.CODE_PREDICTOR: CodePredictorExecutionRow,
            SynthesisStage.CODEC: (CodecExecutionRow, CodecMetadataExecutionRow),
        }[batch.stage]
        expected_types = expected if isinstance(expected, tuple) else (expected,)
        if any(type(row) not in expected_types for row in values.rows):
            raise TypeError(f"{batch.stage.value} received the wrong stage input row")
        if values.requires_host_finalize and batch.stage is not SynthesisStage.TALKER_DECODE:
            raise ValueError("only Talker decode may require host finalize")
        if values.reuse_attention_plan and batch.stage is not SynthesisStage.TALKER_DECODE:
            raise ValueError("only Talker decode may reuse an attention plan")

        if batch.stage is SynthesisStage.TALKER_PREFILL:
            self._preflight_prefill(batch, values)
        elif batch.stage is SynthesisStage.CODE_PREDICTOR:
            if not isinstance(batch.compatibility, CodePredictorBatchCompatibility):
                raise TypeError("Code Predictor batch compatibility is invalid")
        elif batch.stage is SynthesisStage.CODEC:
            self._preflight_codec(batch, values)

    def preflight(
        self,
        decision: ScheduleDecision,
        inputs: tuple[StageExecutionInputs, ...],
    ) -> None:
        if not isinstance(decision, ScheduleDecision):
            raise TypeError("execution preflight requires a ScheduleDecision")
        if not isinstance(inputs, tuple) or len(inputs) != len(decision.batches):
            raise ValueError("execution inputs must match every decision batch")
        for batch, values in zip(decision.batches, inputs, strict=True):
            self._preflight_batch(batch, values)

    @staticmethod
    def _output_device(result: object) -> torch.device | None:
        if isinstance(result, TalkerExecutionResult):
            value = result.result.tokens
        elif isinstance(result, CodePredictorResult):
            value = result.frames
        elif isinstance(result, CodecResult):
            value = result.pcm
        else:
            return None
        return value.device if value.is_cuda else None

    def _record_completion_fence(self, device: torch.device | None) -> CudaSubmissionFence:
        factory = self.completion_event_factory
        if factory is not None:
            if not callable(factory):
                raise TypeError("completion_event_factory must be callable")
            return CudaSubmissionFence(factory())
        if device is None:
            return CudaSubmissionFence.completed()
        return CudaSubmissionFence.record(device=device)

    @staticmethod
    def _record_decision_fence(device: torch.device | None) -> CudaSubmissionFence:
        if device is None:
            return CudaSubmissionFence.completed()
        return CudaSubmissionFence.record(device=device)

    def _submit_batch(
        self,
        batch: StageExecutionBatch,
        inputs: StageExecutionInputs,
    ) -> object:
        if batch.capture is None:
            self.empty_terminal(rows=batch.logical_rows)
            return CodecResult(
                torch.empty((batch.logical_rows, 0), dtype=torch.int16),
                None,
                terminal=True,
            )
        key = batch.capture.key
        if batch.stage is SynthesisStage.TALKER_PREFILL:
            return self.talker_prefill_rows(
                key,
                TalkerPrefillRowsExecutionInput(batch.request_ids, inputs.rows),
            )
        if batch.stage is SynthesisStage.TALKER_DECODE:
            return self.talker_decode_rows(
                key,
                TalkerDecodeRowsExecutionInput(
                    batch.request_ids,
                    inputs.rows,
                    reuse_attention_plan=inputs.reuse_attention_plan,
                ),
            )
        if batch.stage is SynthesisStage.CODE_PREDICTOR:
            compatibility = batch.compatibility
            assert isinstance(compatibility, CodePredictorBatchCompatibility)
            return self.code_predictor_rows(
                key,
                CodePredictorRowsExecutionInput(
                    inputs.rows,
                    sampler_route=compatibility.sampler_route,
                ),
            )
        compatibility = batch.compatibility
        assert isinstance(compatibility, CodecBatchCompatibility)
        return self.codec_rows(
            key,
            CodecRowsExecutionInput(
                inputs.rows,
                visible_frames=compatibility.visible_frames,
                pcm_start_frame=compatibility.pcm_start_frame,
                terminal=compatibility.terminal,
            ),
        )

    def submit(
        self,
        decision: ScheduleDecision,
        inputs: tuple[StageExecutionInputs, ...],
    ) -> tuple[StageExecutionSubmission, ...]:
        self.preflight(decision, inputs)
        return self.submit_preflighted(decision, inputs)

    def submit_preflighted(
        self,
        decision: ScheduleDecision,
        inputs: tuple[StageExecutionInputs, ...],
    ) -> tuple[StageExecutionSubmission, ...]:
        """Submit a decision whose complete input surface was already validated."""
        staged: list[tuple[StageExecutionBatch, object, CudaSubmissionFence, bool]] = []
        decision_device: torch.device | None = None
        for batch, values in zip(decision.batches, inputs, strict=True):
            result = self._submit_batch(batch, values)
            device = self._output_device(result)
            if device is not None:
                decision_device = device
            completion_fence = (
                self._record_completion_fence(device)
                if values.requires_host_finalize
                else CudaSubmissionFence.completed()
            )
            staged.append(
                (batch, result, completion_fence, values.requires_host_finalize)
            )
        decision_fence = self._record_decision_fence(decision_device)
        last = len(staged) - 1
        return tuple(
            StageExecutionSubmission(
                batch=batch,
                result=result,
                completion_fence=completion_fence,
                decision_fence=decision_fence if index == last else None,
                requires_host_finalize=requires_host_finalize,
            )
            for index, (batch, result, completion_fence, requires_host_finalize) in enumerate(staged)
        )


__all__ = ["Executor"]

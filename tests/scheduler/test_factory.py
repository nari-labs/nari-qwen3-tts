from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts.contract import CodecCaptureKey, CodecExecutionMode, EncodedText
from nari_qwen3_tts.contract.errors import RequestRejected
from nari_qwen3_tts.contract.request import SynthesisRequest
from nari_qwen3_tts.engine.admission import make_admitted_request
from nari_qwen3_tts.engine.engine import Engine
from nari_qwen3_tts.executor import (
    CodecExecutionRow,
    CodecRowsExecutionInput,
)
from nari_qwen3_tts.executor.executor import Executor
from nari_qwen3_tts.executor.types import CodecResult
from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState
from nari_qwen3_tts.planner import CaptureCatalog, CaptureCoverageError
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader

from .test_pipeline_loop import _input_plan


def test_runtime_request_uses_profile_schedule_and_measured_silence_guard() -> None:
    request = SynthesisRequest(
        text="default contract",
        voice="aiden",
        language="english",
        non_streaming_mode=True,
    )
    runtime_request = make_admitted_request(
        request_id="request",
        request=request,
        talker_plan=_input_plan(0, streaming=False),
        execution_config=ProfileLoader().load_profile(ExecutionProfile.TTFA),
        admitted_at_s=1.25,
    )

    assert runtime_request.chunk_schedule == (2, 4, 8, 12)
    assert runtime_request.suppress_first_silent_frame

    custom = SynthesisRequest(
        text="custom schedule",
        voice="aiden",
        language="english",
        non_streaming_mode=True,
        stream_chunk_schedule=(3, 7),
    )
    runtime_custom = make_admitted_request(
        request_id="custom",
        request=custom,
        talker_plan=_input_plan(0, streaming=False),
        execution_config=ProfileLoader().load_profile(ExecutionProfile.TTFA),
        admitted_at_s=2.0,
    )
    assert runtime_custom.chunk_schedule == (3, 7)
    assert not runtime_custom.suppress_first_silent_frame


def test_engine_admission_keeps_encoded_text_on_host_until_executor_staging() -> None:
    prepared_devices: list[tuple[str, str]] = []

    class Talker:
        model = SimpleNamespace(device="meta")

        @staticmethod
        def prepare_prepared_inputs(prepared_inputs):
            prepared_devices.append(
                (
                    prepared_inputs[0].text_token_ids.device.type,
                    prepared_inputs[0].instruct_token_ids.device.type,
                )
            )
            return _input_plan(0, streaming=False)

    class Pipeline:
        @staticmethod
        def admit(_request) -> None:
            return None

    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    engine = object.__new__(Engine)
    engine.model = SimpleNamespace(
        prepare=lambda request: EncodedText(
            request,
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
        )
    )
    engine.executor = SimpleNamespace(talker=Talker(), config=config)
    engine.pipeline = Pipeline()
    engine.catalog = CaptureCatalog.from_config(config.stages)

    engine._admit_request(
        "request",
        SynthesisRequest(text="device boundary"),
        admitted_at_s=0.0,
        live=False,
    )

    assert prepared_devices == [("cpu", "cpu")]


def test_runtime_lowering_selects_the_tightest_capture_for_each_batch_membership() -> None:
    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    catalog = CaptureCatalog.from_config(config.stages)

    prefill_single = catalog.lower_talker_prefill((11,))[0]
    prefill_batch = catalog.lower_talker_prefill((11, 17, 9))[0]
    decode_single = catalog.lower_talker_decode(1)[0]
    decode_batch = catalog.lower_talker_decode(4)[0]
    predictor_single = catalog.lower_code_predictor(1)[0]
    predictor_batch = catalog.lower_code_predictor(4)[0]
    codec_single = catalog.lower_codec(
        CodecExecutionMode.WARM,
        model_frames=1,
        logical_rows=1,
    )[0]
    codec_batch = catalog.lower_codec(
        CodecExecutionMode.WARM,
        model_frames=1,
        logical_rows=4,
    )[0]

    # Eleven tokens use the smaller B2/T20 exact-plan capture instead of B1/T32.
    assert prefill_single.capture_batch_size == 2
    assert prefill_batch.capture_batch_size == 4
    assert decode_single.capture_batch_size == 1
    assert decode_batch.capture_batch_size == 4
    assert predictor_single.capture_batch_size == 1
    assert predictor_batch.capture_batch_size == 4
    assert codec_single.capture_batch_size == 1
    assert codec_batch.capture_batch_size == 4


def test_prefill_lowering_splits_at_both_row_and_token_capacity() -> None:
    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    catalog = CaptureCatalog.from_config(config.stages)

    parts = catalog.lower_talker_prefill((400, 400, 200, 200, 200, 200))

    assert [(part.logical_start, part.logical_stop) for part in parts] == [
        (0, 1),
        (1, 3),
        (3, 6),
    ]
    assert [part.capture_batch_size for part in parts] == [1, 2, 4]
    assert all(part.padding == part.capture_batch_size - part.logical_rows for part in parts)


def test_codec_request_schedule_must_be_covered_before_admission() -> None:
    ttfa = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.TTFA).stages
    )
    balanced = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.BALANCED).stages
    )
    throughput = CaptureCatalog.from_config(
        ProfileLoader().load_profile(ExecutionProfile.THROUGHPUT).stages
    )

    ttfa.validate_codec_schedule((1, 2, 4, 8, 12))
    balanced.validate_codec_schedule((4, 4, 8, 16, 25))
    throughput.validate_codec_schedule((25,))
    balanced.validate_codec_schedule((4,))
    with pytest.raises(CaptureCoverageError, match="Codec capture"):
        ttfa.validate_codec_schedule((3, 7))
    with pytest.raises(CaptureCoverageError, match="Codec capture"):
        balanced.validate_codec_schedule((13,))


def test_engine_admission_rejects_uncovered_codec_schedule_before_state_publish() -> None:
    class Talker:
        model = SimpleNamespace(device="cpu")

        @staticmethod
        def prepare_prepared_inputs(prepared):
            plan = _input_plan(0, streaming=False)
            if prepared[0].request.text.startswith("oversized"):
                length = 641
                return replace(
                    plan,
                    text_token_ids=torch.arange(length),
                    codec_token_ids=torch.zeros(length, dtype=torch.long),
                    codec_token_mask=torch.zeros(length, dtype=torch.bool),
                    sequence_lengths=(length,),
                )
            return plan

    class Runtime:
        def __init__(self) -> None:
            self.admitted = []

        def admit(self, request):
            self.admitted.append(request)

    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    runtime = Runtime()
    engine = object.__new__(Engine)
    engine.model = SimpleNamespace(
        prepare=lambda value: EncodedText(
            value,
            torch.tensor([1]),
            torch.empty(0, dtype=torch.long),
        )
    )
    engine.executor = SimpleNamespace(talker=Talker(), config=config)
    engine.pipeline = runtime
    engine.catalog = CaptureCatalog.from_config(config.stages)
    request = SynthesisRequest(
        text="unsupported capture shape",
        stream_chunk_schedule=(13,),
    )
    prepared = EncodedText(request, torch.tensor([1]), torch.empty(0, dtype=torch.long))

    with pytest.raises(RequestRejected, match="Codec capture"):
        engine._admit_request(
            "request",
            prepared.request,
            admitted_at_s=0.0,
            live=False,
        )

    oversized = SynthesisRequest(text="oversized prefill")
    oversized_prepared = EncodedText(
        oversized,
        torch.tensor([1]),
        torch.empty(0, dtype=torch.long),
    )
    with pytest.raises(RequestRejected, match="Talker prefill"):
        engine._admit_request(
            "oversized",
            oversized_prepared.request,
            admitted_at_s=0.0,
            live=False,
        )

    assert runtime.admitted == []


class _CudaGraphRuntime:
    def __init__(self) -> None:
        self.codec_args = None
        self.empty_rows = 0

    def replay_codec(self, key, values):
        self.codec_args = (key, values)
        return CodecResult(torch.ones((1, 2), dtype=torch.int16), None, terminal=False)

    def record_empty_terminal(self, *, rows):
        self.empty_rows = rows


class _CudaRuntimeStub:
    def __init__(self) -> None:
        self.runtime = _CudaGraphRuntime()
        self.added = []
        self.removed = []

    def add_request(self, request_id):
        self.added.append(request_id)

    def remove_request(self, request_id):
        self.removed.append(request_id)


def test_cuda_execution_preserves_direct_codec_row_sources_until_static_staging() -> None:
    cuda_runtime = _CudaRuntimeStub()
    execution = Executor(
        config=None,
        required_keys=None,
        talker=None,
        code_predictor=None,
        codec=SimpleNamespace(replay=cuda_runtime.runtime.replay_codec),
        optimizations=None,
    )
    frame = torch.arange(16)
    state = IncrementalCodecState()
    call = CodecRowsExecutionInput(
        rows=(CodecExecutionRow(frames=(frame,), state=state),),
        visible_frames=1,
    )

    key = CodecCaptureKey(CodecExecutionMode.WHOLE_SEQUENCE, 1, 1)
    output = execution.codec_rows(key, call)

    assert output.pcm.tolist() == [[1, 1]]
    staged = cuda_runtime.runtime.codec_args[1]
    assert isinstance(staged, CodecRowsExecutionInput)
    assert staged is call
    assert staged.rows[0].frames[0] is frame
    assert staged.rows[0].state is state



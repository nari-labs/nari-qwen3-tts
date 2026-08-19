from __future__ import annotations

import copy
import pickle

import pytest


def test_canonical_stage_values_and_lane_mapping_are_fixed() -> None:
    from nari_qwen3_tts.contract.stage import RequestLane, SynthesisStage

    assert tuple(stage.value for stage in SynthesisStage) == (
        "talker_prefill",
        "talker_decode",
        "code_predictor",
        "codec",
    )
    assert SynthesisStage.TALKER_PREFILL.lane is RequestLane.GENERATION
    assert SynthesisStage.TALKER_DECODE.lane is RequestLane.GENERATION
    assert SynthesisStage.CODE_PREDICTOR.lane is RequestLane.GENERATION
    assert SynthesisStage.CODEC.lane is RequestLane.CODEC
    assert len({stage.value for stage in SynthesisStage}) == 4


def test_capture_keys_reject_invalid_values() -> None:
    from nari_qwen3_tts.contract.stage import (
        CodecCaptureKey,
        CodecExecutionMode,
        TalkerDecodeCaptureKey,
    )

    with pytest.raises((TypeError, ValueError)):
        TalkerDecodeCaptureKey(0)
    with pytest.raises((TypeError, ValueError)):
        CodecCaptureKey(CodecExecutionMode.EMPTY, 1, 1)


def test_capture_keys_deepcopy_and_pickle_as_canonical_types() -> None:
    from nari_qwen3_tts.contract.stage import (
        CodecCaptureKey,
        CodecExecutionMode,
    )

    value = CodecCaptureKey(CodecExecutionMode.WARM, 12, 4)
    assert copy.deepcopy(value) == value
    assert pickle.loads(pickle.dumps(value)) == value

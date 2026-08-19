from __future__ import annotations

import pytest


def test_planner_contract_rows_derive_padding() -> None:
    from nari_qwen3_tts.contract.work import (
        TALKER_DECODE_COMPATIBILITY,
        StageBatchRow,
    )

    real = StageBatchRow(0, "request-1", 3, 7, TALKER_DECODE_COMPATIBILITY)
    padding = StageBatchRow(1, None, None, 7, TALKER_DECODE_COMPATIBILITY)

    assert not real.padding
    assert padding.padding
    with pytest.raises(ValueError, match="padding row"):
        StageBatchRow(1, None, 3, 7, TALKER_DECODE_COMPATIBILITY)


def test_stage_execution_batch_binds_stage_capture_and_row_types() -> None:
    from nari_qwen3_tts.contract.stage import (
        CudaGraphRef,
        SynthesisStage,
        TalkerDecodeCaptureKey,
    )
    from nari_qwen3_tts.contract.work import (
        TALKER_DECODE_COMPATIBILITY,
        ScheduleDecision,
        StageBatchRow,
        StageExecutionBatch,
        TalkerPrefillBatchCompatibility,
    )

    capture = CudaGraphRef(
        SynthesisStage.TALKER_DECODE,
        TalkerDecodeCaptureKey(capture_batch_size=2),
    )
    batch = StageExecutionBatch(
        batch_id=4,
        decision_id=3,
        stage=SynthesisStage.TALKER_DECODE,
        compatibility=TALKER_DECODE_COMPATIBILITY,
        capture=capture,
        rows=(
            StageBatchRow(0, "request-1", 2, 5, TALKER_DECODE_COMPATIBILITY),
            StageBatchRow(1, None, None, 5, TALKER_DECODE_COMPATIBILITY),
        ),
    )

    assert batch.request_ids == ("request-1",)
    assert batch.logical_rows == 1
    assert batch.padding_rows == 1
    assert ScheduleDecision(3, (batch,)).batches == (batch,)

    with pytest.raises(ValueError, match="requires TalkerDecodeBatchCompatibility"):
        StageExecutionBatch(
            batch_id=4,
            decision_id=3,
            stage=SynthesisStage.TALKER_DECODE,
            compatibility=TalkerPrefillBatchCompatibility(5),
            capture=capture,
            rows=(StageBatchRow(0, "request-1", 2, 5, TalkerPrefillBatchCompatibility(5)),),
        )


def test_codec_batch_compatibility_preserves_shape_rules_with_typed_mode() -> None:
    from nari_qwen3_tts.contract.stage import CodecExecutionMode
    from nari_qwen3_tts.contract.work import (
        CodecBatchCompatibility,
        codec_batch_compatible,
    )

    first = CodecBatchCompatibility(CodecExecutionMode.WARM, 12, 8, 8, 0, 8, True)
    second = CodecBatchCompatibility(CodecExecutionMode.WARM, 12, 4, 4, 0, 4, True)
    assert codec_batch_compatible(first, second)

    cold = CodecBatchCompatibility(CodecExecutionMode.COLD, 12, 8, 8, 0, 8, True)
    assert not codec_batch_compatible(first, cold)
    with pytest.raises(ValueError, match="metadata"):
        CodecBatchCompatibility(CodecExecutionMode.EMPTY, 0, 0, 0, 0, 0, False)


def test_ready_stage_work_derives_lane_and_validates_deadline_state() -> None:
    from nari_qwen3_tts.contract.stage import RequestLane, SynthesisStage
    from nari_qwen3_tts.contract.work import (
        ReadyStageWork,
        TalkerPrefillBatchCompatibility,
    )

    work = ReadyStageWork(
        request_id="request-1",
        stage=SynthesisStage.TALKER_PREFILL,
        version=0,
        logical_step=0,
        compatibility=TalkerPrefillBatchCompatibility(7),
        admission_sequence=1,
        startup=True,
    )
    assert work.lane is RequestLane.GENERATION
    assert work.identity[0] == "request-1"

    with pytest.raises(ValueError, match="incoherent"):
        ReadyStageWork(
            request_id="request-1",
            stage=SynthesisStage.TALKER_PREFILL,
            version=0,
            logical_step=0,
            compatibility=TalkerPrefillBatchCompatibility(7),
            admission_sequence=1,
            startup=True,
            deadline_s=1.0,
        )

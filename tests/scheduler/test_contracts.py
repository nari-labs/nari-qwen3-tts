from __future__ import annotations

import pytest

from nari_qwen3_tts.contract import (
    TALKER_DECODE_COMPATIBILITY,
    CodePredictorBatchCompatibility,
    CudaGraphRef,
    StageBatchRow,
    StageExecutionBatch,
    SynthesisStage,
    TalkerDecodeCaptureKey,
)


def _real_row(*, physical_row: int = 0, request_id: str = "request-a") -> StageBatchRow:
    return StageBatchRow(
        physical_row=physical_row,
        request_id=request_id,
        version=3,
        logical_step=7,
        compatibility=TALKER_DECODE_COMPATIBILITY,
    )


def test_execution_batch_retains_exact_real_and_padding_row_mapping() -> None:
    rows = (
        _real_row(),
        StageBatchRow(
            physical_row=1,
            request_id=None,
            version=None,
            logical_step=7,
            compatibility=TALKER_DECODE_COMPATIBILITY,
        ),
    )
    batch = StageExecutionBatch(
        batch_id=9,
        decision_id=4,
        stage=SynthesisStage.TALKER_DECODE,
        compatibility=TALKER_DECODE_COMPATIBILITY,
        capture=CudaGraphRef(
            SynthesisStage.TALKER_DECODE,
            TalkerDecodeCaptureKey(2),
        ),
        rows=rows,
    )

    assert batch.logical_rows == 1
    assert batch.capture is not None
    assert batch.capture.key.capture_batch_size == 2
    assert batch.padding_rows == 1
    assert batch.request_ids == ("request-a",)
    assert batch.real_rows == rows[:1]
    assert batch.rows[1].padding


@pytest.mark.parametrize(
    ("request_id", "version"),
    ((None, 0), ("", 0)),
)
def test_rows_can_never_resolve_ambiguous_request_state(
    request_id: str | None,
    version: int | None,
) -> None:
    with pytest.raises(ValueError, match="padding|request"):
        StageBatchRow(
            physical_row=0,
            request_id=request_id,
            version=version,
            logical_step=0,
            compatibility=TALKER_DECODE_COMPATIBILITY,
        )


def test_execution_batch_rejects_misaligned_or_wrong_stage_rows() -> None:
    capture = CudaGraphRef(
        SynthesisStage.TALKER_DECODE,
        TalkerDecodeCaptureKey(1),
    )
    with pytest.raises(ValueError, match="physical rows"):
        StageExecutionBatch(
            batch_id=1,
            decision_id=1,
            stage=SynthesisStage.TALKER_DECODE,
            compatibility=TALKER_DECODE_COMPATIBILITY,
            capture=capture,
            rows=(_real_row(physical_row=1),),
        )
    with pytest.raises(ValueError, match="requires"):
        StageExecutionBatch(
            batch_id=1,
            decision_id=1,
            stage=SynthesisStage.TALKER_DECODE,
            compatibility=TALKER_DECODE_COMPATIBILITY,
            capture=capture,
            rows=(
                StageBatchRow(
                    physical_row=0,
                    request_id="request-a",
                    version=0,
                    logical_step=0,
                    compatibility=CodePredictorBatchCompatibility(),
                ),
            ),
        )

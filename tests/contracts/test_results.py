from __future__ import annotations

import pytest
import torch


def test_stage_completion_retains_token_free_rows_and_typed_deltas() -> None:
    from nari_qwen3_tts.contract.result import (
        StageBatchRowResult,
        StageExecutionCompletion,
        TalkerStateDelta,
    )
    from nari_qwen3_tts.contract.stage import SynthesisStage
    from nari_qwen3_tts.contract.work import (
        TALKER_DECODE_COMPATIBILITY,
        StageBatchRow,
    )

    real = StageBatchRow(0, "request-1", 2, 5, TALKER_DECODE_COMPATIBILITY)
    padding = StageBatchRow(1, None, None, 5, TALKER_DECODE_COMPATIBILITY)
    delta = TalkerStateDelta(
        token=torch.tensor([7], dtype=torch.long),
        hidden=torch.zeros(4),
        logits=torch.zeros(8),
        next_seen_token_mask=torch.zeros(8, dtype=torch.bool),
        next_sampling_offset=32,
        kv=None,
    )
    completion = StageExecutionCompletion(
        batch_id=3,
        stage=SynthesisStage.TALKER_DECODE,
        rows=(StageBatchRowResult(real, delta), StageBatchRowResult(padding, None)),
    )

    assert completion.error is None
    assert completion.rows[0].delta is delta
    with pytest.raises(ValueError, match="padding"):
        StageBatchRowResult(padding, delta)
    with pytest.raises(ValueError, match="real row"):
        StageBatchRowResult(real, None)


def test_failed_completion_is_typed_and_carries_no_partial_rows() -> None:
    from nari_qwen3_tts.contract.result import StageExecutionCompletion
    from nari_qwen3_tts.contract.stage import SynthesisStage

    failure = RuntimeError("capture replay failed")
    completion = StageExecutionCompletion(
        batch_id=3,
        stage=SynthesisStage.CODEC,
        rows=(),
        error=failure,
    )
    assert completion.error is failure

    with pytest.raises(ValueError, match="failed completion"):
        StageExecutionCompletion(
            batch_id=3,
            stage=SynthesisStage.CODEC,
            rows=(),
            error="not typed",  # type: ignore[arg-type]
        )


def test_state_delta_validations_fail_closed() -> None:
    from nari_qwen3_tts.contract.result import (
        CodecStateDelta,
        KVPublication,
        TalkerStateDelta,
    )

    with pytest.raises(ValueError, match="non-empty request ID"):
        KVPublication("", (), 0)
    with pytest.raises(ValueError, match="unique"):
        KVPublication("request-1", (3, 3), 1)
    with pytest.raises(ValueError, match="sampling offset"):
        TalkerStateDelta(
            torch.tensor([1]),
            torch.zeros(2),
            torch.zeros(4),
            torch.zeros(4, dtype=torch.bool),
            -1,
            None,
        )
    with pytest.raises(ValueError, match="visible frames"):
        CodecStateDelta(None, consumed_frames=1, visible_frames=2, terminal=False)

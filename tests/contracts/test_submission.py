from __future__ import annotations

import pytest
import torch


class _Event:
    def __init__(self, *, ready: bool = False) -> None:
        self.is_ready = ready
        self.queries = 0
        self.waits = 0

    def query(self) -> bool:
        self.queries += 1
        return self.is_ready

    def synchronize(self) -> None:
        self.waits += 1
        self.is_ready = True


def test_submission_fence_polls_and_waits_only_its_recorded_event() -> None:
    from nari_qwen3_tts.executor.cuda_graph import CudaSubmissionFence

    event = _Event()
    fence = CudaSubmissionFence(event)

    assert fence.ready() is False
    assert event.queries == 1 and event.waits == 0
    fence.wait()
    assert event.waits == 1
    assert fence.ready() is True

    completed = CudaSubmissionFence.completed()
    assert completed.ready() is True
    completed.wait()


def test_stage_execution_inputs_reject_empty_or_mixed_stage_rows() -> None:
    from nari_qwen3_tts.executor.rows import (
        CodePredictorExecutionRow,
        TalkerDecodeExecutionRow,
        TalkerSamplingExecutionRow,
    )
    from nari_qwen3_tts.executor.submission import StageExecutionInputs

    sampling = TalkerSamplingExecutionRow(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=1,
        offset=0,
        seen_token_mask=None,
    )
    decode = TalkerDecodeExecutionRow(
        talker_step_embed=torch.zeros(4),
        text_token_id=torch.tensor(1),
        suppress_eos=True,
        sampling=sampling,
    )
    predictor = CodePredictorExecutionRow(
        layer0_token=torch.tensor(1),
        past_hidden=torch.zeros(4),
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=1,
        offsets=tuple(range(15)),
    )

    with pytest.raises(ValueError, match="rows"):
        StageExecutionInputs(batch_id=1, rows=())
    with pytest.raises(ValueError, match="one stage input type"):
        StageExecutionInputs(batch_id=1, rows=(decode, predictor))
    with pytest.raises(ValueError, match="host finalize"):
        StageExecutionInputs(
            batch_id=1,
            rows=(predictor,),
            requires_host_finalize=True,
        )

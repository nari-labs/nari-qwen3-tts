from __future__ import annotations

from types import SimpleNamespace

import torch


def _execution_input():
    from nari_qwen3_tts.contract.rng import CodePredictorSamplerRoute
    from nari_qwen3_tts.executor.rows import (
        CodePredictorExecutionRow,
        CodePredictorRowsExecutionInput,
    )

    row = CodePredictorExecutionRow(
        layer0_token=torch.tensor([3], dtype=torch.long),
        past_hidden=torch.zeros(4),
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        seed=7,
        offsets=tuple(range(15)),
    )
    return CodePredictorRowsExecutionInput(
        rows=(row,),
        sampler_route=CodePredictorSamplerRoute.GENERAL,
    )


def _execution_output():
    from nari_qwen3_tts.executor.types import CodePredictorResult

    return CodePredictorResult(
        frames=torch.zeros((1, 16), dtype=torch.long),
        codec_sum=torch.zeros((1, 4)),
    )


class _GraphRuntime:
    def __init__(self, output) -> None:
        self.output = output
        self.call = None

    def replay_code_predictor(self, key, values):
        self.call = (key, values)
        return self.output


def _cuda_execution(runtime):
    from nari_qwen3_tts.executor.executor import Executor

    return Executor(
        None,
        None,
        None,
        SimpleNamespace(replay=runtime.replay_code_predictor),
        None,
        None,
    )


def test_code_predictor_dispatch_forwards_input_and_result_without_allocating() -> None:
    from nari_qwen3_tts.contract import CodePredictorCaptureKey

    values = _execution_input()
    output = _execution_output()
    runtime = _GraphRuntime(output)
    execution = _cuda_execution(runtime)
    key = CodePredictorCaptureKey(1)

    returned = execution.code_predictor_rows(key, values)

    assert runtime.call == (key, values)
    assert runtime.call[1] is values
    assert runtime.call[1].rows[0] is values.rows[0]
    assert returned is output
    assert returned.frames.untyped_storage().data_ptr() == output.frames.untyped_storage().data_ptr()
    assert returned.codec_sum.untyped_storage().data_ptr() == output.codec_sum.untyped_storage().data_ptr()

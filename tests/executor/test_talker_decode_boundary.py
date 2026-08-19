from __future__ import annotations

from types import SimpleNamespace

import torch


def _decode_input():
    from nari_qwen3_tts.executor.rows import (
        TalkerDecodeExecutionRow,
        TalkerDecodeRowsExecutionInput,
        TalkerSamplingExecutionRow,
    )

    sampling = TalkerSamplingExecutionRow(0.9, 50, 1.0, 1.05, 7, 512, None)
    row = TalkerDecodeExecutionRow(
        talker_step_embed=torch.zeros(4),
        text_token_id=torch.tensor([3], dtype=torch.long),
        suppress_eos=False,
        sampling=sampling,
    )
    return TalkerDecodeRowsExecutionInput(
        request_ids=("request-1",),
        rows=(row,),
        reuse_attention_plan=True,
    )


def _decode_output():
    from nari_qwen3_tts.executor.rows import TalkerExecutionResult
    from nari_qwen3_tts.executor.types import TalkerResult

    return TalkerExecutionResult(
        result=TalkerResult(
            tokens=torch.tensor([5], dtype=torch.long),
            last_hidden=torch.zeros((1, 4)),
            logits=torch.zeros((1, 8)),
        ),
        next_seen_token_masks=torch.zeros((1, 8), dtype=torch.bool),
        next_sampling_offsets=torch.tensor([1024], dtype=torch.long),
        kv_publications=(_Publication(),),
    )


class _Publication:
    def validate(self) -> None:
        return None

    def publish(self) -> None:
        return None

    def discard(self) -> None:
        return None


class _GraphRuntime:
    def __init__(self, output) -> None:
        self.output = output
        self.call = None

    def replay_talker_decode(self, key, values):
        self.call = (key, values)
        return self.output


def _cuda_execution(runtime):
    from nari_qwen3_tts.executor.executor import Executor

    return Executor(
        None,
        None,
        SimpleNamespace(replay=runtime.replay_talker_decode),
        None,
        None,
        None,
    )


def test_decode_dispatch_forwards_input_and_result_without_allocating() -> None:
    from nari_qwen3_tts.contract import TalkerDecodeCaptureKey

    values = _decode_input()
    output = _decode_output()
    runtime = _GraphRuntime(output)
    execution = _cuda_execution(runtime)
    key = TalkerDecodeCaptureKey(1)

    returned = execution.talker_decode_rows(key, values)

    assert runtime.call == (key, values)
    assert runtime.call[1] is values
    assert runtime.call[1].rows[0] is values.rows[0]
    assert runtime.call[1].reuse_attention_plan is True
    assert returned is output
    assert returned.result.tokens.untyped_storage().data_ptr() == output.result.tokens.untyped_storage().data_ptr()

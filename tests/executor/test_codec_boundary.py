from __future__ import annotations

from types import SimpleNamespace

import torch


def _execution_input():
    from nari_qwen3_tts.executor.rows import CodecExecutionRow, CodecRowsExecutionInput
    from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState

    frame = torch.arange(16, dtype=torch.long)
    state = IncrementalCodecState()
    row = CodecExecutionRow(
        frames=(frame,),
        state=state,
        visible_frames=1,
        pcm_start_frame=0,
        terminal=False,
    )
    return CodecRowsExecutionInput(
        rows=(row,),
        visible_frames=1,
        pcm_start_frame=0,
        terminal=False,
    )


def _execution_output():
    from nari_qwen3_tts.executor.types import CodecResult
    from nari_qwen3_tts.model.incremental_codec import IncrementalCodecState

    return CodecResult(
        pcm=torch.zeros((1, 2), dtype=torch.int16),
        states=(IncrementalCodecState(frame_position=1, transformer_context_length=1),),
        terminal=False,
        pcm_lengths=(2,),
    )


class _GraphRuntime:
    def __init__(self, output) -> None:
        self.output = output
        self.call = None

    def replay_codec(self, key, values):
        self.call = (key, values)
        return self.output


def _cuda_execution(runtime):
    from nari_qwen3_tts.executor.executor import Executor

    return Executor(
        None,
        None,
        None,
        None,
        SimpleNamespace(replay=runtime.replay_codec),
        None,
    )


def test_codec_dispatch_forwards_input_and_result_without_allocating() -> None:
    from nari_qwen3_tts.contract import CodecCaptureKey, CodecExecutionMode

    values = _execution_input()
    output = _execution_output()
    runtime = _GraphRuntime(output)
    execution = _cuda_execution(runtime)
    key = CodecCaptureKey(CodecExecutionMode.WHOLE_SEQUENCE, 1, 1)

    returned = execution.codec_rows(key, values)

    assert runtime.call == (key, values)
    assert runtime.call[1] is values
    assert runtime.call[1].rows[0] is values.rows[0]
    assert runtime.call[1].rows[0].state is values.rows[0].state
    assert runtime.call[1].rows[0].frames[0] is values.rows[0].frames[0]
    assert returned is output
    assert returned.pcm.untyped_storage().data_ptr() == output.pcm.untyped_storage().data_ptr()
    assert returned.states is output.states

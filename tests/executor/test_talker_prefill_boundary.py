from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _prefill_input():
    from nari_qwen3_tts.executor.rows import (
        TalkerPrefillExecutionRow,
        TalkerPrefillRowsExecutionInput,
        TalkerSamplingExecutionRow,
    )

    sampling = TalkerSamplingExecutionRow(0.9, 50, 1.0, 1.05, 7, 0, None)
    row = TalkerPrefillExecutionRow(
        text_token_ids=torch.tensor([1, 2], dtype=torch.long),
        codec_token_ids=torch.tensor([3, 4], dtype=torch.long),
        codec_token_mask=torch.tensor([False, True]),
        suppress_eos=True,
        sampling=sampling,
    )
    return TalkerPrefillRowsExecutionInput(("request-1",), (row,))


def _prefill_output():
    from nari_qwen3_tts.executor.rows import TalkerExecutionResult
    from nari_qwen3_tts.executor.types import TalkerResult

    return TalkerExecutionResult(
        result=TalkerResult(
            tokens=torch.tensor([5], dtype=torch.long),
            last_hidden=torch.zeros((1, 4)),
            logits=torch.zeros((1, 8)),
        ),
        next_seen_token_masks=torch.zeros((1, 8), dtype=torch.bool),
        next_sampling_offsets=torch.tensor([512], dtype=torch.long),
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

    def replay_talker_prefill(self, key, values):
        self.call = (key, values)
        return self.output


def _cuda_execution(runtime):
    from nari_qwen3_tts.executor.executor import Executor

    return Executor(
        None,
        None,
        SimpleNamespace(replay=runtime.replay_talker_prefill),
        None,
        None,
        None,
    )


def test_prefill_dispatch_forwards_input_and_result_without_allocating() -> None:
    from nari_qwen3_tts.contract import TalkerPrefillCaptureKey

    values = _prefill_input()
    output = _prefill_output()
    runtime = _GraphRuntime(output)
    execution = _cuda_execution(runtime)
    key = TalkerPrefillCaptureKey(1, 2, 2)

    returned = execution.talker_prefill_rows(key, values)

    assert runtime.call == (key, values)
    assert runtime.call[1] is values
    assert runtime.call[1].rows[0] is values.rows[0]
    assert returned is output
    assert returned.result.tokens.untyped_storage().data_ptr() == output.result.tokens.untyped_storage().data_ptr()


def test_real_kv_publication_owns_fail_closed_validation() -> None:
    from nari_qwen3_tts.executor.cache import PagedKVAllocator

    allocator = PagedKVAllocator(total_pages=8, page_size=4)
    allocator.add_request("request-1")
    publication = allocator.reserve(("request-1",), (2,)).publications[0]

    publication.validate()
    publication.publish()
    with pytest.raises(RuntimeError, match="stale|consumed|pending"):
        publication.validate()

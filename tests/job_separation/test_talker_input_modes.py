from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts.contract import EncodedText, TextContinuation
from nari_qwen3_tts.contract.request import SynthesisRequest
from nari_qwen3_tts.executor.input_layout import (
    TalkerInputMode,
    TalkerInputPlan,
    prepare_talker_input_plan,
)
from nari_qwen3_tts.executor.talker import TalkerExecutor as TalkerCudaExecutor


def _config():
    predictor = SimpleNamespace(vocab_size=64)
    talker = SimpleNamespace(
        spk_id={"aiden": 17},
        spk_is_dialect={"aiden": False},
        codec_language_id={"english": 23},
        codec_think_id=30,
        codec_nothink_id=31,
        codec_think_bos_id=32,
        codec_think_eos_id=33,
        codec_pad_id=34,
        codec_bos_id=35,
        codec_eos_token_id=63,
        hidden_size=1,
        vocab_size=128,
        code_predictor=predictor,
    )
    return SimpleNamespace(
        tts_bos_token_id=90,
        tts_eos_token_id=91,
        tts_pad_token_id=92,
        talker=talker,
        talker_config=talker,
    )


class _Model:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.embedding = torch.nn.Embedding(128, 1)

    @staticmethod
    def initialize_projected_text_embedding_cache() -> None:
        return None

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding


def _text(offset: int = 0) -> torch.Tensor:
    return torch.tensor([1, 2, 3, 10, 11, 12, 13, 80, 81, 82, 83, 84]) + offset


def _plan(modes: list[bool]):
    rows = len(modes)
    return prepare_talker_input_plan(
        _config(),
        text_inputs=[_text(index) for index in range(rows)],
        instruct_inputs=[torch.empty(0, dtype=torch.long) for _ in range(rows)],
        languages=["english"] * rows,
        speakers=["aiden"] * rows,
        non_streaming_modes=modes,
    )


def test_full_text_prefill_then_pad_decode() -> None:
    plan = _plan([True])
    assert plan.modes == (TalkerInputMode.FULL_TEXT_PREFILL,)
    assert plan.sequence_lengths == (15,)
    assert plan.text_token_ids[-6:].tolist() == [10, 11, 12, 13, 91, 92]
    assert [plan.continuations[0].token_at(step).item() for step in range(4)] == [92] * 4


def test_first_text_token_prefill_then_target_eos_pad_decode() -> None:
    plan = _plan([False])
    assert plan.modes == (TalkerInputMode.STREAMING_DECODE,)
    assert plan.sequence_lengths == (10,)
    assert plan.text_token_ids[-1:].tolist() == [10]
    assert [plan.continuations[0].token_at(step).item() for step in range(6)] == [11, 12, 13, 91, 92, 92]


def test_mixed_batch_keeps_row_local_mode_and_step() -> None:
    plan = _plan([False, True])
    assert plan.modes == (TalkerInputMode.STREAMING_DECODE, TalkerInputMode.FULL_TEXT_PREFILL)
    assert plan.decode_text_tokens((0, 0)).tolist() == [11, 92]
    assert plan.decode_text_tokens((3, 7)).tolist() == [91, 92]


def test_nonempty_instruction_is_preserved_before_each_mode_specific_prefill() -> None:
    instruction = torch.tensor([40, 41])
    plan = prepare_talker_input_plan(
        _config(),
        text_inputs=[_text(), _text(1)],
        instruct_inputs=[instruction, instruction + 2],
        languages=["english", "english"],
        speakers=["aiden", "aiden"],
        non_streaming_modes=[False, True],
    )

    assert plan.sequence_lengths == (12, 17)
    first_row_end = plan.sequence_lengths[0]
    assert plan.text_token_ids[:2].tolist() == [40, 41]
    assert plan.codec_token_mask[:2].tolist() == [False, False]
    assert plan.text_token_ids[first_row_end : first_row_end + 2].tolist() == [42, 43]
    assert plan.codec_token_mask[first_row_end : first_row_end + 2].tolist() == [False, False]
    assert plan.decode_text_tokens((0, 0)).tolist() == [11, 92]


def test_talker_job_consumes_prepared_request_contract() -> None:
    prepared = [
        EncodedText(
            request=SynthesisRequest(
                text="streaming",
                voice="Aiden",
                language="English",
                non_streaming_mode=False,
            ),
            text_token_ids=_text(0),
            instruct_token_ids=torch.empty(0, dtype=torch.long),
        ),
        EncodedText(
            request=SynthesisRequest(
                text="whole",
                voice="Aiden",
                language="English",
                non_streaming_mode=True,
            ),
            text_token_ids=_text(1),
            instruct_token_ids=torch.empty(0, dtype=torch.long),
        ),
    ]
    job = TalkerCudaExecutor(
        model=_Model(),
        config=_config(),
        cache=object(),
        capture_slots=1,
        driver=object(),
    )

    plan = job.prepare_prepared_inputs(prepared)

    assert plan.modes == (TalkerInputMode.STREAMING_DECODE, TalkerInputMode.FULL_TEXT_PREFILL)
    assert plan.decode_text_tokens((0, 0)).tolist() == [11, 92]


def test_input_plan_rejects_invalid_rows_and_steps() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        prepare_talker_input_plan(
            _config(),
            text_inputs=[_text()],
            instruct_inputs=[torch.empty(0, dtype=torch.long)],
            languages=["english"],
            speakers=["aiden"],
            non_streaming_modes=[],
        )
    plan = _plan([False])
    with pytest.raises(ValueError, match="generation-step rows"):
        plan.decode_text_tokens(())
    with pytest.raises(ValueError, match="non-negative"):
        plan.decode_text_tokens((-1,))
    with pytest.raises(TypeError, match="integer"):
        plan.decode_text_tokens((True,))


def test_input_plan_rejects_untyped_or_cross_device_continuations() -> None:
    with pytest.raises(TypeError, match="non_streaming_mode"):
        TextContinuation(
            non_streaming_mode="streaming_decode",  # type: ignore[arg-type]
            token_ids=torch.tensor([1]),
            pad_token_id=torch.tensor([2]),
        )
    with pytest.raises(TypeError, match="long"):
        TextContinuation(
            non_streaming_mode=False,
            token_ids=torch.tensor([1.0]),
            pad_token_id=torch.tensor([2.0]),
        )

    cpu = TextContinuation(
        non_streaming_mode=False,
        token_ids=torch.tensor([1]),
        pad_token_id=torch.tensor([2]),
    )
    meta = TextContinuation(
        non_streaming_mode=True,
        token_ids=torch.tensor([1], device="meta"),
        pad_token_id=torch.tensor([2], device="meta"),
    )
    with pytest.raises(ValueError, match="device"):
        TalkerInputPlan(
            text_token_ids=torch.tensor([1, 2]),
            codec_token_ids=torch.tensor([3, 4]),
            codec_token_mask=torch.tensor([False, True]),
            sequence_lengths=(1, 1),
            continuations=(cpu, meta),
        )


def test_raw_input_plan_rejects_non_integer_text_contract() -> None:
    with pytest.raises(TypeError, match="long"):
        prepare_talker_input_plan(
            _config(),
            text_inputs=[_text().float()],
            instruct_inputs=[torch.empty(0, dtype=torch.long)],
            languages=["english"],
            speakers=["aiden"],
            non_streaming_modes=[False],
        )

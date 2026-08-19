from __future__ import annotations

import copy
import pickle

import pytest
import torch


def test_text_continuation_preserves_live_append_semantics() -> None:
    from nari_qwen3_tts.contract.request import TextContinuation

    continuation = TextContinuation(
        non_streaming_mode=False,
        token_ids=torch.tensor([11], dtype=torch.long),
        pad_token_id=torch.tensor([0], dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([12], dtype=torch.long),
    )

    assert continuation.has_token(0)
    assert not continuation.has_token(1)
    with pytest.raises(RuntimeError, match="has no token"):
        continuation.token_at(1)

    updated = continuation.append(
        torch.tensor([13], dtype=torch.long),
        sequence=0,
        is_final=True,
    )
    assert updated.input_finished
    assert updated.next_update_sequence == 1
    assert updated.materialized_token_ids().tolist() == [11, 13, 12]
    assert updated.token_at(3).tolist() == [0]


@pytest.mark.parametrize(
    "args",
    [
        ((-1,), (), 1),
        ((True,), (), 1),
        ((1.5,), (), 1),
        ((1,), (), 0),
    ],
)
def test_fragment_tokenization_rejects_invalid_token_ownership(args) -> None:
    from nari_qwen3_tts.contract.request import FragmentTokenization

    with pytest.raises((TypeError, ValueError), match="token|consum"):
        FragmentTokenization(*args)


def test_live_append_does_not_recopy_prior_cuda_tokens(monkeypatch) -> None:
    from nari_qwen3_tts.contract.request import TextContinuation

    continuation = TextContinuation(
        non_streaming_mode=False,
        token_ids=torch.tensor([11], dtype=torch.long),
        pad_token_id=torch.tensor([0], dtype=torch.long),
        input_finished=False,
        terminal_token_id=torch.tensor([12], dtype=torch.long),
    )
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live append must not concatenate the accumulated token history")
        ),
    )

    updated = continuation.append(
        torch.tensor([13, 14], dtype=torch.long),
        sequence=0,
        is_final=True,
    )

    assert updated.token_count == 4
    assert [updated.token_at(step).item() for step in range(4)] == [11, 13, 14, 12]


def test_request_family_preserves_validation_pickle_and_deepcopy() -> None:
    from nari_qwen3_tts.contract.model import SynthesisModelSpec
    from nari_qwen3_tts.contract.request import SynthesisRequest

    request = SynthesisRequest(text="hello", voice=" Aiden ", language="ENGLISH")
    assert request.voice == "aiden"
    assert request.language == "english"
    assert request.effective_max_output_tokens == 2_048
    assert copy.deepcopy(request) == request
    assert pickle.loads(pickle.dumps(request)) == request

    spec = SynthesisModelSpec(codec_eos_token_id=31, talker_vocab_size=32)
    assert copy.deepcopy(spec) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec

    with pytest.raises(ValueError, match="exactly 16 codebooks"):
        SynthesisModelSpec(codec_eos_token_id=31, talker_vocab_size=32, num_codebooks=15)

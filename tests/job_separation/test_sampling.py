from __future__ import annotations

import pytest
import torch

from nari_qwen3_tts.executor.talker import TalkerExecutor as TalkerCudaExecutor
from nari_qwen3_tts.executor.types import TalkerSamplingInput
from nari_qwen3_tts.model.sampling import sample_logits_stateless


def test_sampling_is_row_local_replayable_and_offset_sensitive() -> None:
    logits = torch.zeros((2, 32))
    kwargs = {
        "temperature": torch.ones(2),
        "top_k": torch.full((2,), 32, dtype=torch.int32),
        "top_p": torch.ones(2),
        "seed": torch.tensor([41, 42]),
        "offset": torch.tensor([0, 6]),
    }
    first = sample_logits_stateless(logits, **kwargs)
    second = sample_logits_stateless(logits, **kwargs)
    torch.testing.assert_close(first, second)
    changed = sample_logits_stateless(logits, **{**kwargs, "offset": torch.tensor([1, 6])})
    assert first[0] != changed[0]
    assert first[1] == changed[1]


def test_sampling_matches_singletons_and_request_rows_survive_permutation() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.7, -0.3],
            [0.7, 0.2, 0.1, -0.1],
            [-0.2, 0.8, 0.3, 0.4],
        ]
    )
    kwargs = {
        "temperature": torch.tensor([0.7, 1.1, 0.9]),
        "top_k": torch.tensor([4, 3, 2], dtype=torch.int32),
        "top_p": torch.tensor([1.0, 0.9, 0.8]),
        "seed": torch.tensor([41, 42, 43]),
        "offset": torch.tensor([5, 6, 7]),
    }
    batched = sample_logits_stateless(logits, **kwargs)
    singleton = torch.cat(
        [
            sample_logits_stateless(
                logits[row : row + 1],
                **{name: value[row : row + 1] for name, value in kwargs.items()},
            )
            for row in range(logits.shape[0])
        ]
    )
    torch.testing.assert_close(batched, singleton)

    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    permuted = sample_logits_stateless(
        logits.index_select(0, permutation),
        **{name: value.index_select(0, permutation) for name, value in kwargs.items()},
    )
    torch.testing.assert_close(batched, permuted.index_select(0, inverse))


def test_greedy_rows_do_not_consume_global_rng() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    tokens = sample_logits_stateless(
        torch.tensor([[1.0, 3.0, 2.0]]),
        temperature=torch.zeros(1),
        top_k=torch.zeros(1, dtype=torch.int32),
        top_p=torch.ones(1),
        seed=torch.tensor([99]),
        offset=torch.tensor([100]),
    )
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert tokens.tolist() == [1]


def test_greedy_talker_still_applies_row_local_repetition_penalty() -> None:
    values = TalkerSamplingInput(
        temperature=torch.zeros(2),
        top_k=torch.zeros(2, dtype=torch.int32),
        top_p=torch.ones(2),
        repetition_penalty=torch.tensor([2.0, 1.0]),
        seed=torch.tensor([31, 32]),
        offsets=torch.tensor([7, 8]),
        seen_token_mask=torch.tensor([[True, False], [False, False]]),
    )
    logits = torch.tensor([[10.0, 9.0], [10.0, 9.0]])

    tokens = TalkerCudaExecutor._sample_direct(logits, values)

    assert tokens.tolist() == [1, 0]


def test_all_greedy_talker_cannot_bypass_sampling_domain_validation() -> None:
    values = TalkerSamplingInput(
        temperature=torch.zeros(1),
        top_k=torch.tensor([-1], dtype=torch.int32),
        top_p=torch.ones(1),
        repetition_penalty=torch.ones(1),
        seed=torch.tensor([31]),
        offsets=torch.tensor([7]),
    )

    with pytest.raises(ValueError, match="top_k"):
        TalkerCudaExecutor._sample_direct(torch.tensor([[10.0, 9.0]]), values)


def test_mixed_greedy_and_sampled_rows_preserve_private_rng_and_singleton_meaning() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0], [0.0, 0.0, 0.0]])
    kwargs = {
        "temperature": torch.tensor([0.0, 0.8]),
        "top_k": torch.tensor([0, 3], dtype=torch.int32),
        "top_p": torch.tensor([1.0, 1.0]),
        "repetition_penalty": torch.tensor([2.0, 1.0]),
        "seen_token_mask": torch.tensor([[True, False, False], [False, False, False]]),
        "seed": torch.tensor([71, 72]),
        "offset": torch.tensor([9, 10]),
    }
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()

    batch = sample_logits_stateless(logits, **kwargs)
    singletons = torch.cat(
        [
            sample_logits_stateless(
                logits[row : row + 1],
                **{name: value[row : row + 1] for name, value in kwargs.items()},
            )
            for row in range(2)
        ]
    )

    torch.testing.assert_close(batch, singletons)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert batch[0].item() == 1


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("temperature", torch.tensor([float("nan")]), "temperature"),
        ("top_k", torch.tensor([-1], dtype=torch.int32), "top_k"),
        ("top_p", torch.tensor([float("inf")]), "top_p"),
        ("seed", torch.tensor([-1]), "seed"),
        ("offset", torch.tensor([-1]), "offset"),
        ("repetition_penalty", torch.tensor([float("inf")]), "repetition_penalty"),
    ],
)
def test_sampling_rejects_nonfinite_or_negative_logical_parameters(
    field: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    values = {
        "temperature": torch.ones(1),
        "top_k": torch.ones(1, dtype=torch.int32),
        "top_p": torch.ones(1),
        "seed": torch.ones(1, dtype=torch.long),
        "offset": torch.ones(1, dtype=torch.long),
        "repetition_penalty": torch.ones(1),
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=message):
        sample_logits_stateless(torch.zeros((1, 4)), **values)


def test_sampling_does_not_mutate_logits_or_request_parameters() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
    values = {
        "temperature": torch.tensor([0.8, 1.1]),
        "top_k": torch.tensor([0, 2], dtype=torch.int32),
        "top_p": torch.tensor([0.9, 0.8]),
        "seed": torch.tensor([41, 42]),
        "offset": torch.tensor([5, 6]),
        "repetition_penalty": torch.tensor([1.2, 1.0]),
        "seen_token_mask": torch.tensor([[True, False, False], [False, False, False]]),
    }
    logits_snapshot = logits.clone()
    snapshots = {name: value.clone() for name, value in values.items()}

    sample_logits_stateless(logits, **values)

    torch.testing.assert_close(logits, logits_snapshot)
    for name, value in values.items():
        torch.testing.assert_close(value, snapshots[name])


def test_direct_sampling_requires_exact_tensor_kinds() -> None:
    with pytest.raises(TypeError, match="top_k"):
        sample_logits_stateless(
            torch.zeros((1, 4)),
            temperature=torch.ones(1),
            top_k=torch.ones(1),
            top_p=torch.ones(1),
            seed=torch.ones(1, dtype=torch.long),
            offset=torch.ones(1, dtype=torch.long),
        )

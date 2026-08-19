from __future__ import annotations

import torch

from nari_qwen3_tts.executor.sampling import (
    sample_code_predictor_cuda_graph,
    sample_logits_cuda_graph,
)


def test_capture_sampler_keeps_greedy_and_row_local_stochastic_inputs_explicit() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 1.0]])
    tokens = sample_logits_cuda_graph(
        logits,
        temperature=torch.zeros(2),
        top_k=torch.ones(2, dtype=torch.int32),
        top_p=torch.ones(2),
        seed=torch.tensor([7, 11]),
        offset=torch.tensor([0, 32]),
    )
    assert tokens.tolist() == [1, 0]


def test_code_predictor_sampler_has_a_graph_safe_greedy_cpu_reference() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 1.0]])
    tokens = sample_code_predictor_cuda_graph(
        logits,
        temperature=torch.zeros(2),
        top_k=torch.ones(2, dtype=torch.int32),
        top_p=torch.ones(2),
        seed=torch.tensor([7, 11]),
        offset=torch.tensor([0, 32]),
    )
    assert tokens.tolist() == [1, 0]



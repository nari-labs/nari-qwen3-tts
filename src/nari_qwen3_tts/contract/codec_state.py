"""Request-local tensor state shared by Codec model and execution layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class IncrementalCodecState:
    """Request-local causal state for one Codec stream."""

    frame_position: int = 0
    transformer_context_length: int = 0
    transformer_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    transformer_values: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_overlaps: dict[str, torch.Tensor] = field(default_factory=dict)


__all__ = ["IncrementalCodecState"]

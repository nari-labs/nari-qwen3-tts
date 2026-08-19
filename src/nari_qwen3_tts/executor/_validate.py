"""Scalar ingress checks shared by every typed execution row.

Each executor validates the per-request scalars it stages into static buffers.
The rule is written once here so Talker, Code Predictor, and Codec cannot drift
into accepting different values for the same field.
"""

from __future__ import annotations

import math


def require_scalar_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def require_scalar_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


__all__ = ["require_scalar_int", "require_scalar_number"]


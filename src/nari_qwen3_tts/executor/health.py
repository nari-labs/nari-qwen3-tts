"""Fail-closed capture startup and execution accounting contracts."""

from __future__ import annotations


class CaptureStartupError(RuntimeError):
    pass


class UncapturedExecutionError(RuntimeError):
    pass


__all__ = [
    "CaptureStartupError",
    "UncapturedExecutionError",
]

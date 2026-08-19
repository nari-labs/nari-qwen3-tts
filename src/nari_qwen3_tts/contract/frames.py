"""Shared Codec frame-lifecycle constants."""

from __future__ import annotations

# The whole-sequence window decoded before incremental Codec state exists.
WHOLE_SEQUENCE_MAX_FRAMES = 3

# The cold template used to initialize warm incremental state.
WARM_TEMPLATE_FRAMES = WHOLE_SEQUENCE_MAX_FRAMES + 4

# Valid unpadded terminal-cold windows that may be padded to a captured shape.
COLD_TERMINAL_PAD_FRAMES = tuple(
    range(WHOLE_SEQUENCE_MAX_FRAMES + 1, WARM_TEMPLATE_FRAMES + 1)
)

__all__ = [
    "COLD_TERMINAL_PAD_FRAMES",
    "WHOLE_SEQUENCE_MAX_FRAMES",
    "WARM_TEMPLATE_FRAMES",
]

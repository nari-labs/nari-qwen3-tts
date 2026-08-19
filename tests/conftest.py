from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def syntax_items(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield every syntax item without relying on source-text matching."""

    pending = [tree]
    while pending:
        item = pending.pop()
        yield item
        for field in item._fields:
            value = getattr(item, field)
            if isinstance(value, ast.AST):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(element for element in value if isinstance(element, ast.AST))

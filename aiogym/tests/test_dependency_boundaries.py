from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _imports(package: str) -> set[str]:
    imports: set[str] = set()
    root = PACKAGE_ROOT / package
    for path in root.rglob("*.py"):
        module_parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = max(0, len(module_parts) - node.level + 1)
                    prefix = ("aiogym", *module_parts[:keep])
                    resolved = ".".join(
                        (*prefix, *((node.module or "").split(".")))
                    ).rstrip(".")
                else:
                    resolved = node.module or ""
                imports.add(resolved)
    return imports


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("models", ("aiogym.evaluation", "aiogym.controllers", "aiogym.rl")),
        ("rewards", ("aiogym.evaluation",)),
    ],
)
def test_dependency_direction(package, forbidden):
    imports = _imports(package)
    violations = sorted(
        imported
        for imported in imports
        if any(
            imported == boundary or imported.startswith(boundary + ".")
            for boundary in forbidden
        )
    )
    assert not violations

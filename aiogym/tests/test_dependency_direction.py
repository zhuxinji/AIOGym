from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


CORE = Path(__file__).resolve().parents[1] / "core"
FORBIDDEN = {"scenarios", "controllers", "workflows", "cli", "rl"}


def test_core_does_not_import_higher_layers():
    violations = []
    for path in CORE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[:2] == ["aiogym", "core"]:
                        continue
                    if alias.name.startswith("aiogym."):
                        violations.append((path.name, alias.name))
            if module.startswith("aiogym."):
                layer = module.split(".", 2)[1]
                if layer in FORBIDDEN:
                    violations.append((path.name, module))
    assert violations == []


def test_importing_core_does_not_load_optional_dependencies():
    code = (
        "import sys; import aiogym.core; "
        "blocked=('torch','stable_baselines3','casadi','onnx','onnxruntime','optuna'); "
        "loaded=[name for name in blocked if name in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

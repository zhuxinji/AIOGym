from __future__ import annotations

import json
import subprocess
import sys

from aiogym.evaluation.provenance import runtime_environment


def test_runtime_environment_is_json_safe_and_explicit():
    value = runtime_environment(
        vector_backend="dummy",
        multiprocessing_start_method="spawn",
    )
    json.dumps(value, allow_nan=False)
    assert value["vector_backend"] == "dummy"
    assert value["multiprocessing_start_method"] == "spawn"
    assert value["python_version"]
    assert value["aiogym_version"]


def test_provenance_support_does_not_import_optional_frameworks():
    code = """
import sys
import aiogym
from aiogym.evaluation.provenance import runtime_environment
runtime_environment()
unexpected = {'torch', 'stable_baselines3', 'casadi', 'onnx', 'onnxruntime'} & set(sys.modules)
assert not unexpected, sorted(unexpected)
"""
    subprocess.run([sys.executable, "-c", code], check=True)

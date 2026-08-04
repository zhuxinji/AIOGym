"""Capture reproducibility-relevant runtime facts without modifying them."""
from __future__ import annotations

import argparse
import importlib.metadata
import multiprocessing
import os
import platform
import subprocess
import sys

from aiogym._internal.serialization import write_json_artifact


RUNTIME_ENVIRONMENT_SCHEMA_VERSION = "aiogym.runtime_environment.v1"


def capture_runtime_environment() -> dict:
    versions = {
        name: _version(name)
        for name in (
            "aiogym",
            "numpy",
            "gymnasium",
            "torch",
            "stable-baselines3",
            "casadi",
        )
    }
    torch_runtime = _torch_runtime()
    return {
        "schema_version": RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependency_versions": versions,
        "torch_runtime": torch_runtime,
        "cpu_count": os.cpu_count(),
        "multiprocessing_start_method": multiprocessing.get_start_method(
            allow_none=True
        ),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_runtime() -> dict:
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "available": False,
            "cuda_available": False,
            "cuda_version": None,
            "cudnn_version": None,
            "device_name": None,
            "num_threads": None,
            "num_interop_threads": None,
        }
    cuda_available = bool(torch.cuda.is_available())
    return {
        "available": True,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": (
            torch.cuda.get_device_name(0) if cuda_available else "cpu"
        ),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_json_artifact(args.output, capture_runtime_environment())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

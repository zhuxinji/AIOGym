#!/usr/bin/env python3
"""Capture pre-refactor AIO-Gym behavior outside the source tree."""
from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/tmp/aiogym-refactor-baseline")
SCENARIOS = {
    "quadruple": ("minimum-phase", "regulation-v1"),
    "cascade": ("continuous-benchmark", "regulation-v1"),
    "cascade-recirculating": (
        "commissioning",
        "regulation-v1",
    ),
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=REPO_ROOT, text=True
    ).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(payload),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _python_files(*, include_tests: bool) -> list[Path]:
    files = sorted((REPO_ROOT / "aiogym").rglob("*.py"))
    if include_tests:
        return files
    return [path for path in files if "tests" not in path.parts]


def _size_snapshot() -> dict[str, int]:
    all_files = _python_files(include_tests=True)
    source_files = _python_files(include_tests=False)
    return {
        "python_files": len(all_files),
        "python_lines": sum(
            len(path.read_text(encoding="utf-8").splitlines())
            for path in all_files
        ),
        "non_test_python_files": len(source_files),
        "non_test_python_lines": sum(
            len(path.read_text(encoding="utf-8").splitlines())
            for path in source_files
        ),
    }


def _dependency_graph() -> dict[str, list[str]]:
    package_root = REPO_ROOT / "aiogym"
    top_level = {
        path.name
        for path in package_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    graph: dict[str, set[str]] = {name: set() for name in top_level}
    for path in _python_files(include_tests=False):
        relative = path.relative_to(package_root)
        source = relative.parts[0] if len(relative.parts) > 1 else "__root__"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("aiogym."):
                        target = alias.name.split(".", 2)[1]
                        if source in graph and target in top_level and target != source:
                            graph[source].add(target)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("aiogym."):
                    target = module.split(".", 2)[1]
                elif node.level and len(relative.parts) > 1:
                    base = list(relative.parts[:-1])
                    keep = max(0, len(base) - node.level + 1)
                    parts = base[:keep] + ([module.split(".")[0]] if module else [])
                    target = parts[0] if parts else None
                if source in graph and target in top_level and target != source:
                    graph[source].add(target)
    return {name: sorted(targets) for name, targets in sorted(graph.items())}


def _trajectory(scenario: str, case: str, reward: str) -> dict:
    from aiogym import make_env

    env = make_env(
        config={
            "scenario": scenario,
            "case": case,
            "reward_spec": reward,
            "info_level": "full",
            "environment": {"episode_steps": 5},
        }
    )
    try:
        observation, reset_info = env.reset(seed=1729)
        action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        rows = []
        for step in range(5):
            next_observation, scalar_reward, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            rows.append(
                {
                    "step": step,
                    "observation": observation,
                    "state": np.asarray(env.integ.x, dtype=float),
                    "reward": float(scalar_reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "next_observation": next_observation,
                    "info": info,
                }
            )
            observation = next_observation
            if terminated or truncated:
                break
        return {
            "scenario": scenario,
            "case": case,
            "reward_spec": reward,
            "seed": 1729,
            "action": action,
            "observation_space": {
                "low": env.observation_space.low,
                "high": env.observation_space.high,
            },
            "action_space": {
                "low": env.action_space.low,
                "high": env.action_space.high,
            },
            "reset_info": reset_info,
            "rows": rows,
        }
    finally:
        env.close()


def _controller_rollouts() -> dict[str, Any]:
    from aiogym import make_controller, make_env
    from aiogym.evaluation.execution.rollouts import rollout_controller

    output = {}
    for controller_id in ("pid", "mpc"):
        env = make_env(
            config={
                "scenario": "quadruple",
                "case": "minimum-phase",
                "reward_spec": "regulation-v1",
                "info_level": "full",
                "environment": {"episode_steps": 5},
            }
        )
        try:
            controller = make_controller(
                controller_id,
                scenario="quadruple",
                config={"P": 2} if controller_id == "mpc" else None,
            )
            output[controller_id] = rollout_controller(
                controller, env, seed=2718, max_steps=5
            )
        finally:
            env.close()
    return output


def _design_result() -> dict[str, Any]:
    from aiogym.design import run_design_study

    example = REPO_ROOT / "configs/design/cascade-recirculating-example-v1.json"
    return run_design_study(example, robustness_samples=2, seed=7)


def _dataset_v2(output: Path) -> dict[str, Any]:
    from aiogym.datasets.collect import collect_dataset
    from aiogym.datasets.config import DatasetCollectionConfig
    from aiogym.datasets.reader import DatasetReader, validate_dataset

    dataset_path = output / "dataset-v2"
    config = DatasetCollectionConfig(
        {
            "schema_version": "aiogym.dataset_collection.v1",
            "track_id": "quadruple-regulation-generalist-v1",
            "dataset_id": "core-refactor-baseline-v2",
            "split": "training",
            "base_seed": 31415,
            "target_transitions": 1,
            "workers": 1,
            "collectors": [{"id": "nominal_pid", "weight": 1.0}],
            "output": str(dataset_path),
        }
    )
    summary = collect_dataset(config)
    reader = DatasetReader(dataset_path, verify_checksums=True)
    episode = reader.load_episode(0)
    return {
        "path": str(dataset_path),
        "summary": summary,
        "integrity": validate_dataset(dataset_path),
        "manifest": reader.manifest,
        "first_episode": {
            "metadata": episode.metadata,
            "content_hash": episode.content_hash,
            "array_shapes": {
                name: list(value.shape)
                for name, value in episode.arrays().items()
            },
        },
    }


def _sac_smoke(output: Path) -> dict[str, Any]:
    from aiogym.rl import RLTrainingConfig, run_experiment

    result = run_experiment(
        RLTrainingConfig(
            track_id="quadruple-regulation-generalist-v1",
            algorithm_id="sac",
            total_transitions=2,
            training_seed=1618,
            n_envs=1,
            device="cpu",
            validation_seeds=(1619,),
            algorithm={
                "batch_size": 2,
                "torch_threads": 1,
                "utd_ratio": 1.0,
                "vector_backend": "dummy",
                "verbose": 0,
            },
            replay={"capacity": 16, "learning_starts": 1},
            evaluation={"every_transitions": 0},
            output={
                "directory": str(output / "sac-smoke"),
                "name": "core-refactor-baseline-sac",
                "save_rollout": False,
            },
        )
    )
    return {
        "algorithm_id": result.algorithm_id,
        "policy_path": result.policy_path,
        "validation": result.validation,
        "artifact_check": result.artifact_check,
    }


def capture(output: Path, *, with_sac: bool) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"baseline output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = _git("status", "--short").splitlines()
    _write_json(
        output / "repository.json",
        {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty_status": status,
            "size": _size_snapshot(),
            "dependency_graph": _dependency_graph(),
            "python": sys.version,
        },
    )
    for scenario, (case, reward) in SCENARIOS.items():
        _write_json(
            output / f"trajectory-{scenario}.json",
            _trajectory(scenario, case, reward),
        )
    _write_json(output / "controller-rollouts.json", _controller_rollouts())
    _write_json(output / "three-tank-design-result.json", _design_result())
    _write_json(output / "dataset-v2-summary.json", _dataset_v2(output))
    if with_sac:
        _write_json(output / "sac-smoke.json", _sac_smoke(output))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--with-sac", action="store_true")
    parser.add_argument("--sac-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.sac_only:
        result_path = output / "sac-smoke.json"
        if not output.is_dir():
            raise FileNotFoundError(f"baseline output does not exist: {output}")
        if result_path.exists():
            raise FileExistsError(f"SAC baseline already exists: {result_path}")
        _write_json(result_path, _sac_smoke(output))
    else:
        capture(output, with_sac=bool(args.with_sac))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

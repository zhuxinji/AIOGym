"""Descriptively compare policies on the same ordered seeds."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiogym.core.io import write_json

from ._comparison_svg import render_trajectory_svg
from .evaluate import evaluate


COMPARISON_SCHEMA_VERSION = "aiogym.comparison.v5"
TRAJECTORY_ARCHIVE_SCHEMA_VERSION = "aiogym.trajectory-archive.v1"
_TRAJECTORY_FIELDS = (
    "physical_time",
    "true_state",
    "output",
    "reference",
    "commanded_action",
    "channel_action",
    "applied_action",
    "reward",
    "disturbance",
    "constraint_costs",
    "minimum_safety_margin",
)

def compare_policies(
    *,
    env,
    policies: Mapping[str, Any],
    seeds: Sequence[int],
    max_steps: int | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Compare policies and persist one compact report plus trajectory archive.

    Benchmark environments default to
    ``runs/<scenario>/benchmarks/<benchmark>`` and
    replace only the three managed comparison artifacts. Comparisons outside a
    Benchmark require an explicit empty output directory.
    """
    if not isinstance(policies, Mapping) or len(policies) < 2:
        raise ValueError("policies must be a mapping with at least two entries")
    labels = tuple(policies)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("policy labels must be non-empty strings")
    output_directory, overwrite = _prepare_output_directory(env, output)

    evaluations = {
        label: evaluate(
            env=env,
            policy=policies[label],
            seeds=seeds,
            max_steps=max_steps,
        )
        for label in labels
    }
    first = evaluations[labels[0]]
    ordered_seeds = first["seeds"]
    if any(result["seeds"] != ordered_seeds for result in evaluations.values()):
        raise ValueError("all policy evaluations must use identical ordered seeds")
    ranking_metrics = first["ranking_metrics"]
    if any(
        result["ranking_metrics"] != ranking_metrics for result in evaluations.values()
    ):
        raise ValueError("all policy evaluations must use identical ranking metrics")
    first_schema = first["trajectory_schema"]
    static_schema_fields = (
        "time_unit",
        "state",
        "output",
        "action",
        "disturbance_names",
    )
    if any(
        any(
            result["trajectory_schema"][field] != first_schema[field]
            for field in static_schema_fields
        )
        for result in evaluations.values()
    ):
        raise ValueError("all policy evaluations must use compatible trajectory schemas")
    trajectory_schema = {
        **first_schema,
        "constraint_cost_names": sorted(
            {
                name
                for result in evaluations.values()
                for name in result["trajectory_schema"]["constraint_cost_names"]
            }
        ),
    }

    def ranking_key(label):
        result = evaluations[label]
        values = []
        for metric in ranking_metrics:
            value = result["aggregate"][metric["name"]][metric["aggregate"]]
            values.append(value if metric["direction"] == "minimize" else -value)
        return (*values, label)

    ordering = sorted(labels, key=ranking_key)
    trajectory_seed = ordered_seeds[0]
    archive = _write_trajectory_archive(
        output_directory / "trajectories.npz",
        evaluations,
        trajectory_schema,
        overwrite=overwrite,
    )
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "environment": dict(first["environment"]),
        "trajectory_schema": trajectory_schema,
        "seeds": list(ordered_seeds),
        "trajectory_seed": trajectory_seed,
        "trajectory_archive": archive,
        "max_steps": first["max_steps"],
        "ranking_metrics": ranking_metrics,
        "ordering": ordering,
        "evaluations": {
            label: _compact_evaluation(evaluations[label], trajectory_seed)
            for label in labels
        },
    }
    svg_path = output_directory / "comparison.svg"
    svg_path.write_text(render_trajectory_svg(result), encoding="utf-8")
    write_json(
        output_directory / "comparison.json",
        result,
        overwrite=overwrite,
    )
    return result


def _prepare_output_directory(env, output) -> tuple[Path, bool]:
    overwrite = output is None
    directory = _default_output_directory(env) if overwrite else Path(output)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"comparison output is not a directory: {directory}")
    entries = tuple(directory.iterdir()) if directory.exists() else ()
    if not overwrite and entries:
        raise FileExistsError(
            f"refusing to create comparison in non-empty directory: {directory}"
        )
    managed_names = {"comparison.json", "comparison.svg", "trajectories.npz"}
    invalid = sorted(
        path.name
        for path in entries
        if path.name in managed_names
        and (path.is_symlink() or not path.is_file())
    )
    if invalid:
        raise FileExistsError(
            f"default comparison artifacts must be files: {invalid}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory, overwrite


def _compact_evaluation(evaluation, trajectory_seed):
    episodes = []
    for episode in evaluation["episodes"]:
        compact_episode = {
            key: value for key, value in episode.items() if key != "trajectory"
        }
        if episode["seed"] == trajectory_seed:
            compact_episode["trajectory"] = episode["trajectory"]
        episodes.append(compact_episode)
    return {
        **{
            key: value
            for key, value in evaluation.items()
            if key not in {"episodes", "trajectory_summary"}
        },
        "episodes": episodes,
    }


def _write_trajectory_archive(path, evaluations, schema, *, overwrite):
    target = Path(path)
    arrays = {}
    entries = []
    columns = _trajectory_columns(schema)
    for policy_index, (label, evaluation) in enumerate(evaluations.items()):
        for episode_index, episode in enumerate(evaluation["episodes"]):
            prefix = f"p{policy_index:03d}_e{episode_index:03d}"
            trajectory_arrays = _trajectory_arrays(
                episode["trajectory"],
                columns,
            )
            for field, array in trajectory_arrays.items():
                arrays[f"{prefix}__{field}"] = array
            entries.append(
                {
                    "id": prefix,
                    "policy": label,
                    "seed": episode["seed"],
                    "length": episode["length"],
                }
            )
    _write_npz(target, arrays, overwrite=overwrite)
    return {
        "schema_version": TRAJECTORY_ARCHIVE_SCHEMA_VERSION,
        "file": target.name,
        "dtype": "float64",
        "fields": list(_TRAJECTORY_FIELDS),
        "columns": columns,
        "entries": entries,
    }


def _trajectory_columns(schema):
    state = [row["name"] for row in schema["state"]]
    output = [row["name"] for row in schema["output"]]
    action = [row["name"] for row in schema["action"]]
    return {
        "true_state": state,
        "output": output,
        "reference": output,
        "commanded_action": action,
        "channel_action": action,
        "applied_action": action,
        "disturbance": list(schema["disturbance_names"]),
        "constraint_costs": list(schema["constraint_cost_names"]),
    }


def _trajectory_arrays(trajectory, columns):
    missing = set(_TRAJECTORY_FIELDS) - set(trajectory)
    if missing:
        raise ValueError(f"trajectory fields are missing: {sorted(missing)}")
    lengths = {field: len(trajectory[field]) for field in _TRAJECTORY_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"trajectory fields have inconsistent lengths: {lengths}")
    length = next(iter(lengths.values()))
    arrays = {}
    scalar_fields = {
        "physical_time",
        "reward",
        "minimum_safety_margin",
    }
    mapping_fields = {"disturbance", "constraint_costs"}
    for field in _TRAJECTORY_FIELDS:
        if field in scalar_fields:
            array = np.asarray(trajectory[field], dtype=np.float64)
            expected_shape = (length,)
        elif field in mapping_fields:
            names = columns[field]
            rows = trajectory[field]
            unexpected = sorted(
                {
                    name
                    for row in rows
                    for name in row
                    if name not in names
                }
            )
            if unexpected:
                raise ValueError(
                    f"trajectory field {field!r} contains unknown columns: {unexpected}"
                )
            if field == "disturbance":
                missing_names = sorted(
                    {
                        name
                        for row in rows
                        for name in names
                        if name not in row
                    }
                )
                if missing_names:
                    raise ValueError(
                        f"trajectory field {field!r} is missing columns: {missing_names}"
                    )
            array = np.asarray(
                [
                    [float(row[name]) if name in row else 0.0 for name in names]
                    for row in rows
                ],
                dtype=np.float64,
            ).reshape(length, len(names))
            expected_shape = (length, len(names))
        else:
            array = np.asarray(trajectory[field], dtype=np.float64)
            expected_shape = (length, len(columns[field]))
        if array.shape != expected_shape:
            raise ValueError(
                f"trajectory field {field!r} must have shape {expected_shape}; "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"trajectory field {field!r} must contain finite values")
        arrays[field] = array
    return arrays


def _write_npz(path, arrays, *, overwrite):
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_output_directory(env) -> Path:
    base_env = env.unwrapped
    if base_env.benchmark is None:
        raise ValueError(
            "output is required when comparing policies outside a benchmark"
        )
    scenario_id = base_env.scenario.id.replace("_", "-")
    benchmark_id = base_env.benchmark.id
    return Path("runs") / scenario_id / "benchmarks" / benchmark_id


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "compare_policies",
    "render_trajectory_svg",
]

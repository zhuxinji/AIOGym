"""Export evaluation reports without changing their in-memory results."""

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from aiogym.core.io import write_json

from ._comparison_svg import render_trajectory_svg


COMPARISON_SCHEMA_VERSION = "aiogym.comparison.v7"
TRAJECTORY_ARCHIVE_SCHEMA_VERSION = "aiogym.trajectory-archive.v1"
_COMPARISON_FILES = ("trajectories.npz", "comparison.svg", "comparison.json")
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

def _write_evaluation_report(result, output_directory):
    pending = output_directory / ".comparison-pending"
    pending.mkdir()
    cleanup = True
    try:
        archive = _write_trajectory_archive(
            pending / "trajectories.npz",
            result["evaluations"],
            result["trajectory_schema"],
        )
        report = {
            **result,
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "trajectory_archive": archive,
            "evaluations": {
                label: _compact_evaluation(evaluation, result["trajectory_seed"])
                for label, evaluation in result["evaluations"].items()
            },
        }
        svg = render_trajectory_svg(report)
        ET.fromstring(svg)
        (pending / "comparison.svg").write_text(svg, encoding="utf-8")
        write_json(pending / "comparison.json", report, overwrite=False)

        cleanup = False
        attempted = []
        committed = False
        try:
            for name in _COMPARISON_FILES:
                target = output_directory / name
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
                attempted.append(name)
                os.replace(pending / name, target)
            committed = True
        finally:
            if not committed:
                try:
                    for name in reversed(attempted):
                        (output_directory / name).unlink(missing_ok=True)
                except OSError as error:
                    raise OSError(
                        f"evaluation rollback failed; recovery files retained at {pending}"
                    ) from error
            cleanup = True
    finally:
        if cleanup:
            shutil.rmtree(pending)


def _prepare_output_directory(output) -> Path:
    directory = Path(output)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"evaluation output is not a directory: {directory}")
    pending = directory / ".comparison-pending"
    if pending.exists():
        raise FileExistsError(
            f"evaluation write is active or unfinished: {pending}; "
            "inspect the pending directory before retrying"
        )
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to create evaluation in non-empty directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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


def _write_trajectory_archive(path, evaluations, schema):
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
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
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

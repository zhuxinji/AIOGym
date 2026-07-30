"""Streaming quality and coverage reports for Dataset v2."""
from __future__ import annotations

import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .reader import DatasetReader


QUALITY_REPORT_SCHEMA_VERSION = "aiogym.dataset_quality.v1"


def build_quality_report(
    dataset: DatasetReader | str | Path,
    *,
    sample_limit: int = 100_000,
) -> dict:
    """Scan a dataset with bounded memory and summarize integrity/coverage."""

    reader = (
        dataset if isinstance(dataset, DatasetReader) else DatasetReader(dataset)
    )
    collectors = Counter()
    terminal_reasons = Counter()
    tags = Counter()
    spec_hashes = Counter()
    quality_tags = Counter()
    parameter_values = defaultdict(list)
    reward_returns = []
    cost_totals = Counter()
    cost_violations = Counter()
    state_min = None
    state_max = None
    sampled_states = []
    sampled_actions = []
    sampled_slew = []
    sampled_count = 0
    nonfinite_count = 0
    out_of_bounds_count = 0

    for record in reader.metadata_records():
        metadata = record["episode"]["metadata"]
        collectors[metadata["collector_id"]] += 1
        quality_tags[metadata["collector_quality_tag"]] += 1
        terminal_reasons[metadata["termination_reason"]] += 1
        spec_hashes[metadata["resolved_hash"]] += 1
        tags.update(metadata.get("difficulty_tags", ()))
        reward_returns.append(float(metadata["summary"]["return"]))
        for name, value in metadata["plant_parameters"].items():
            values = value if isinstance(value, list) else [value]
            parameter_values[name].extend(float(item) for item in values)

        episode = reader.load_episode(record["episode_id"])
        state = episode.array("true_state")
        next_state = episode.array("next_true_state")
        all_state = np.concatenate((state, next_state[-1:]), axis=0)
        finite = np.isfinite(all_state)
        nonfinite_count += int(np.size(finite) - np.count_nonzero(finite))
        episode_min = np.min(all_state, axis=0)
        episode_max = np.max(all_state, axis=0)
        state_min = (
            episode_min
            if state_min is None
            else np.minimum(state_min, episode_min)
        )
        state_max = (
            episode_max
            if state_max is None
            else np.maximum(state_max, episode_max)
        )
        maximum = metadata["plant_parameters"].get("max_level")
        if maximum is not None:
            out_of_bounds_count += int(
                np.count_nonzero(
                    np.logical_or(
                        all_state < 0.0,
                        all_state > float(maximum),
                    )
                )
            )

        action = episode.array("action_policy_normalized")
        slew = np.diff(action, axis=0)
        remaining = max(0, int(sample_limit) - sampled_count)
        if remaining:
            take = min(remaining, all_state.shape[0])
            indices = np.linspace(
                0,
                all_state.shape[0] - 1,
                num=take,
                dtype=np.int64,
            )
            sampled_states.append(all_state[indices])
            action_indices = np.minimum(indices, action.shape[0] - 1)
            sampled_actions.append(action[action_indices])
            if slew.size:
                slew_indices = np.minimum(indices, slew.shape[0] - 1)
                sampled_slew.append(slew[slew_indices])
            sampled_count += take
        for name, values in episode.cost_channels.items():
            total = float(np.sum(values))
            cost_totals[name] += total
            cost_violations[name] += int(np.count_nonzero(values > 0.0))

    state_sample = (
        np.concatenate(sampled_states, axis=0)
        if sampled_states
        else np.empty((0, 0), dtype=np.float32)
    )
    action_sample = (
        np.concatenate(sampled_actions, axis=0)
        if sampled_actions
        else np.empty((0, 0), dtype=np.float32)
    )
    slew_sample = (
        np.concatenate(sampled_slew, axis=0)
        if sampled_slew
        else np.empty((0, 0), dtype=np.float32)
    )
    duplicate_hashes = sorted(
        value for value, count in spec_hashes.items() if count > 1
    )
    return {
        "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "dataset_id": reader.manifest["dataset_id"],
        "split": reader.manifest["split"],
        "episodes": len(reader),
        "transitions": reader.transition_count,
        "collector_composition": dict(sorted(collectors.items())),
        "collector_quality_composition": dict(sorted(quality_tags.items())),
        "return_distribution": _summary(reward_returns),
        "safety": {
            "cost_totals": dict(sorted(cost_totals.items())),
            "violation_counts": dict(sorted(cost_violations.items())),
        },
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "state": {
            "min": [] if state_min is None else state_min.tolist(),
            "max": [] if state_max is None else state_max.tolist(),
            "quantiles": _column_quantiles(state_sample),
            "normalized_coverage": _normalized_coverage(state_sample),
        },
        "action": {
            "saturation_rate": (
                0.0
                if action_sample.size == 0
                else float(np.mean(np.abs(action_sample) >= 0.999))
            ),
            "slew_quantiles": _column_quantiles(np.abs(slew_sample)),
        },
        "parameter_coverage": {
            name: _summary(values)
            for name, values in sorted(parameter_values.items())
        },
        "duplicate_episode_spec_hashes": duplicate_hashes,
        "integrity": {
            "nonfinite_count": nonfinite_count,
            "out_of_bounds_count": out_of_bounds_count,
        },
        "per_tag_counts": dict(sorted(tags.items())),
    }


def write_quality_report(
    dataset: DatasetReader | str | Path,
) -> dict:
    reader = (
        dataset if isinstance(dataset, DatasetReader) else DatasetReader(dataset)
    )
    report = build_quality_report(reader)
    reports_path = reader.path / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(reports_path / "quality.json", report)
    _atomic_json(
        reports_path / "coverage.json",
        {
            "schema_version": report["schema_version"],
            "dataset_id": report["dataset_id"],
            "state": report["state"],
            "action": report["action"],
            "parameter_coverage": report["parameter_coverage"],
            "per_tag_counts": report["per_tag_counts"],
        },
    )
    return report


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "quantiles": {
            str(q): float(np.quantile(array, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }


def _column_quantiles(array):
    if array.size == 0:
        return {}
    return {
        str(q): np.quantile(array, q, axis=0).tolist()
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def _normalized_coverage(array, bins=20):
    if array.size == 0:
        return {"bins": bins, "per_dimension": [], "mean_occupancy": 0.0}
    occupancy = []
    for column in array.T:
        lo = float(np.min(column))
        hi = float(np.max(column))
        if hi <= lo:
            occupancy.append(1.0 / bins)
            continue
        counts, _ = np.histogram(column, bins=bins, range=(lo, hi))
        occupancy.append(float(np.count_nonzero(counts) / bins))
    return {
        "bins": bins,
        "per_dimension": occupancy,
        "mean_occupancy": float(np.mean(occupancy)),
    }


def _atomic_json(path, data):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "QUALITY_REPORT_SCHEMA_VERSION",
    "build_quality_report",
    "write_quality_report",
]

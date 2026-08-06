#!/usr/bin/env python3
"""One-shot Dataset v2 to v3 conversion; v3 runtime stays compatibility-free."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import aiogym.scenarios  # noqa: F401
from aiogym.core import get_task, stable_hash
from aiogym.datasets.reader import DatasetReader as DatasetV2Reader
from aiogym.workflows.dataset import DatasetReader, DatasetWriter


def migrate(source: Path, output: Path) -> dict:
    old = DatasetV2Reader(source, verify_checksums=True)
    if not len(old):
        raise ValueError("Dataset v2 source contains no episodes")
    first = old.load_episode(0)
    first_metadata = first.metadata
    scenario = str(first_metadata["scenario"]).replace("-", "_")
    objective = str(first_metadata.get("goal", "regulation"))
    task_id = f"{scenario}/{objective}"
    task_hash = get_task(task_id).task_hash
    plants = [old.load_episode(index).metadata["plant_parameters"] for index in range(len(old))]
    plant_hash = stable_hash(plants)
    base_seed = int(first_metadata["base_seed"])
    writer = DatasetWriter(
        output,
        dataset_id=str(old.manifest["dataset_id"]),
        task_id=task_id,
        task_hash=task_hash,
        plant_id=f"migrated-{old.manifest['dataset_id']}",
        plant_hash=plant_hash,
        preset="migrated-v2",
        policy={"id": "migrated-v2", "source_schema": "aiogym.dataset.v2"},
        base_seed=base_seed,
        observation_schema={
            "shape": list(first.array("observation").shape[1:]),
            "dtype": str(first.array("observation").dtype),
        },
        action_schema={
            "shape": list(first.array("action_policy_normalized").shape[1:]),
            "dtype": "float32",
            "bounds": [0.0, 1.0],
        },
        legacy_metadata={
            "source_schema": "aiogym.dataset.v2",
            "source_path": str(source.resolve()),
            "source_manifest_hash": old.manifest.get("manifest_hash"),
            "track_id": first_metadata.get("track_id"),
            "distribution_id": first_metadata.get("distribution_id"),
            "reward_spec_id": first_metadata.get("reward_spec_id"),
            "action_note": (
                "v3 action reconstructs the environment action from "
                "0.5 * (action_policy_normalized + 1); legacy physical action "
                "arrays are retained separately"
            ),
        },
    )
    for index in range(len(old)):
        episode = old.load_episode(index)
        normalized = episode.array("action_policy_normalized")
        action = (0.5 * (normalized + 1.0)).astype(np.float32)
        arrays = {
            "observation": episode.array("observation"),
            "action": action,
            "reward": episode.array("reward_scalar"),
            "next_observation": episode.array("next_observation"),
            "terminated": episode.array("terminated"),
            "truncated": episode.array("truncated"),
            "step_index": episode.array("step_index"),
            "physical_time": episode.array("physical_time"),
            "true_state": episode.array("true_state"),
            "reference": episode.array("reference"),
            "commanded_action": action,
            "applied_action": action,
            "reward_terms": episode.array("reward_terms"),
            "constraint_costs": episode.array("cost_channels"),
            "disturbance": episode.array("measured_disturbance"),
            "legacy_commanded_physical": episode.array("action_commanded_physical"),
            "legacy_applied_physical": episode.array("action_applied_physical"),
        }
        metadata = episode.metadata
        writer.append(
            index,
            int(metadata["base_seed"]),
            arrays,
            metadata={
                "reward_term_names": list(episode.reward_term_names),
                "constraint_cost_names": list(episode.cost_channel_names),
                "legacy_metadata": metadata,
            },
        )
    reader = DatasetReader(output, verify_checksums=True)
    return reader.validate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = migrate(args.source, args.output)
    print(
        f"dataset_id={report['dataset_id']} episodes={report['episode_count']} "
        f"transitions={report['transition_count']} ok={report['ok']}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

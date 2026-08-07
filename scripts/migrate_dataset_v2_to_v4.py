#!/usr/bin/env python3
"""One-shot Dataset v2 to v4 conversion; v4 runtime stays compatibility-free."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import aiogym.scenarios  # noqa: F401
from aiogym.core import file_sha256, get_task, stable_hash
from aiogym.workflows.dataset import DatasetReader, DatasetWriter


def migrate(source: Path, output: Path) -> dict:
    source = Path(source)
    old_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest.get("schema_version") != "aiogym.dataset.v2":
        raise ValueError("source manifest is not aiogym.dataset.v2")
    records = list(old_manifest.get("episodes", ()))
    if not records:
        raise ValueError("Dataset v2 source contains no episodes")
    episodes = [_load_v2_episode(source, record) for record in records]
    first_metadata, first_arrays, _ = episodes[0]
    scenario = str(first_metadata["scenario"]).replace("-", "_")
    objective = str(first_metadata.get("goal", "regulation"))
    task_id = f"{scenario}/{objective}"
    task_hash = get_task(task_id).task_hash
    plants = [metadata["plant_parameters"] for metadata, _, _ in episodes]
    plant_hash = stable_hash(plants)
    base_seed = int(first_metadata["base_seed"])
    writer = DatasetWriter(
        output,
        dataset_id=str(old_manifest["dataset_id"]),
        task_id=task_id,
        task_hash=task_hash,
        plant_id=f"migrated-{old_manifest['dataset_id']}",
        plant_hash=plant_hash,
        condition_id="migrated-v2",
        condition_hash=stable_hash({"source": "aiogym.dataset.v2"}),
        interface_hash=stable_hash(
            {
                "observation_shape": list(first_arrays["observation"].shape[1:]),
                "action_shape": list(
                    first_arrays["action_policy_normalized"].shape[1:]
                ),
            }
        ),
        env_hash=stable_hash({"task_hash": task_hash, "plant_hash": plant_hash}),
        policy={"id": "migrated-v2", "source_schema": "aiogym.dataset.v2"},
        policy_training_contract=None,
        target_environment_contract={
            "task_hash": task_hash,
            "plant_hash": plant_hash,
            "condition_hash": stable_hash({"source": "aiogym.dataset.v2"}),
        },
        contract_status="unverified",
        transfer_flags={
            "is_transfer": False,
            "plant_changed": False,
            "condition_changed": False,
        },
        base_seed=base_seed,
        state_schema={"source": "legacy-v2", "fields": []},
        observation_schema={
            "shape": list(first_arrays["observation"].shape[1:]),
            "dtype": str(first_arrays["observation"].dtype),
        },
        action_schema={
            "shape": list(first_arrays["action_policy_normalized"].shape[1:]),
            "dtype": "float32",
            "bounds": [0.0, 1.0],
        },
        legacy_metadata={
            "source_schema": "aiogym.dataset.v2",
            "source_path": str(source.resolve()),
            "source_manifest_hash": old_manifest.get("manifest_hash"),
            "track_id": first_metadata.get("track_id"),
            "distribution_id": first_metadata.get("distribution_id"),
            "reward_spec_id": first_metadata.get("reward_spec_id"),
            "action_note": (
                "v4 action reconstructs the environment action from "
                "0.5 * (action_policy_normalized + 1); legacy physical action "
                "arrays are retained separately"
            ),
        },
    )
    for index, (metadata, episode, record) in enumerate(episodes):
        normalized = episode["action_policy_normalized"]
        action = (0.5 * (normalized + 1.0)).astype(np.float32)
        arrays = {
            "observation": episode["observation"],
            "action": action,
            "reward": episode["reward_scalar"],
            "next_observation": episode["next_observation"],
            "terminated": episode["terminated"],
            "truncated": episode["truncated"],
            "step_index": episode["step_index"],
            "physical_time": episode["physical_time"],
            "true_state": episode["true_state"],
            "reference": episode["reference"],
            "transition_reference": episode["reference"],
            "commanded_action": action,
            "applied_action": action,
            "reward_terms": episode["reward_terms"],
            "constraint_costs": episode["cost_channels"],
            "disturbance": episode["measured_disturbance"],
            "transition_disturbance": episode["measured_disturbance"],
            "legacy_commanded_physical": episode["action_commanded_physical"],
            "legacy_applied_physical": episode["action_applied_physical"],
        }
        writer.append(
            index,
            int(metadata["base_seed"]),
            arrays,
            metadata={
                "reward_term_names": list(record["episode"]["reward_term_names"]),
                "constraint_cost_names": list(record["episode"]["cost_channel_names"]),
                "legacy_metadata": metadata,
            },
        )
    reader = DatasetReader(output, verify_checksums=True)
    return reader.validate()


def _load_v2_episode(source, record):
    path = source / record["shard"]
    declared = record.get("shard_sha256")
    if declared and file_sha256(path) != declared:
        raise ValueError(f"Dataset v2 shard checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {
        "observation", "action_policy_normalized", "reward_scalar",
        "next_observation", "terminated", "truncated", "step_index",
        "physical_time", "true_state", "reference", "reward_terms",
        "cost_channels", "measured_disturbance", "action_commanded_physical",
        "action_applied_physical",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Dataset v2 shard is missing arrays: {sorted(missing)}")
    return dict(record["episode"]["metadata"]), arrays, record


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

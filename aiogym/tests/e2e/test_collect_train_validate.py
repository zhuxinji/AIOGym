from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

from aiogym import load_track
from aiogym.cli.benchmark import main as benchmark_main
from aiogym.cli.evaluate import main as evaluate_main
from aiogym.controllers.checkpoints import (
    checkpoint_sha256,
    learned_policy_spec_for_track,
    load_policy_checkpoint,
)
from aiogym.datasets.config import DatasetCollectionConfig
from aiogym.datasets.reader import DatasetReader, validate_dataset
from aiogym.datasets.collect import collect_dataset
from aiogym.evaluation.artifact import check_benchmark_artifacts
from aiogym.rl import RLTrainingConfig, run_experiment


TRACK_ID = "quadruple-regulation-generalist-v1"


def test_mixed_collect_validate_and_bc_checkpoint(tmp_path):
    dataset_path = tmp_path / "dataset"
    collection = DatasetCollectionConfig(
        {
            "schema_version": "aiogym.dataset_collection.v1",
            "track_id": TRACK_ID,
            "dataset_id": "e2e-mixed-v2",
            "split": "training",
            "base_seed": 2,
            "target_transitions": 1801,
            "workers": 1,
            "collectors": [
                {"id": "nominal_pid", "weight": 1.0},
                {"id": "noisy_pid", "weight": 1.0},
                {"id": "safe_excitation", "weight": 1.0},
            ],
            "output": str(dataset_path),
        }
    )
    result = collect_dataset(collection)
    integrity = validate_dataset(dataset_path)
    records = DatasetReader(dataset_path).metadata_records()

    assert result["actual_transitions"] == 2700
    assert integrity["ok"]
    assert {row["collector_id"] for row in records} == {
        "nominal_pid",
        "noisy_pid",
        "safe_excitation",
    }
    assert len({row["resolved_hash"] for row in records}) == 3

    trained = run_experiment(
        RLTrainingConfig(
            track_id=TRACK_ID,
            algorithm_id="bc",
            training_seed=31,
            total_transitions=2,
            n_envs=1,
            algorithm={"batch_size": 8, "hidden": 16},
            output={
                "directory": str(tmp_path / "bc"),
                "name": "bc-e2e",
            },
            dataset_id="e2e-mixed-v2",
            dataset_path=str(dataset_path),
            validation_seeds=(7001,),
        )
    )
    digest = checkpoint_sha256(trained.policy_path)
    spec = learned_policy_spec_for_track(
        trained.policy_path,
        "bc",
        digest,
        load_track(TRACK_ID),
    )
    controller = load_policy_checkpoint(spec)
    artifact_dir = Path(trained.artifact_dir)
    training = json.loads(
        (artifact_dir / "data" / "training.json").read_text(
            encoding="utf-8"
        )
    )

    assert controller.metadata()["native_action_contract"] == (
        "normalized[-1,1]"
    )
    assert controller.metadata()["checkpoint"]["sha256"] == digest
    assert trained.validation["split"] == "validation"
    assert trained.artifact_check["ok"]
    assert training["training_config_hash"] == trained.config_hash
    assert training["budget_unit"] == "optimizer_updates"
    assert training["budget_value"] == 2
    assert training["environment_transitions"] == 0
    assert training["optimizer_updates"] == 2
    assert training["offline_samples_available"] == 2700
    assert training["offline_samples_drawn"] > 0
    assert check_benchmark_artifacts(artifact_dir)["ok"]

    evaluate_output = tmp_path / "evaluate.json"
    assert evaluate_main(
        [
            "--checkpoint",
            trained.policy_path,
            "--track",
            TRACK_ID,
            "--algorithm",
            "bc",
            "--sha256",
            digest,
            "--seeds",
            "7001",
            "--output",
            str(evaluate_output),
        ]
    ) == 0
    benchmark_output = tmp_path / "benchmark.json"
    assert benchmark_main(
        [
            TRACK_ID,
            "--controllers",
            "pid",
            "--checkpoint",
            trained.policy_path,
            "--algorithm",
            "bc",
            "--sha256",
            digest,
            "--name",
            "bc",
            "--seeds",
            "7001",
            "--output",
            str(benchmark_output),
        ]
    ) == 0
    evaluated = json.loads(evaluate_output.read_text(encoding="utf-8"))
    benchmarked = json.loads(
        benchmark_output.read_text(encoding="utf-8")
    )["results"]["bc"]
    assert evaluated["aggregate"] == trained.validation["aggregate"]
    assert benchmarked["aggregate"] == trained.validation["aggregate"]

    hybrid_config = RLTrainingConfig(
        track_id=TRACK_ID,
        algorithm_id="rlpd",
        training_seed=33,
        total_transitions=1,
        n_envs=1,
        algorithm={
            "batch_size": 2,
            "bc_steps": 0,
            "pretrain_updates": 1,
            "utd_ratio": 1,
            "n_critics": 2,
            "offline_fraction": 0.5,
        },
        replay={"capacity": 16},
        evaluation={"every_transitions": 100},
        output={
            "directory": str(tmp_path / "rlpd"),
            "name": "rlpd-e2e",
            "save_rollout": False,
        },
        dataset_id="e2e-mixed-v2",
        dataset_path=str(dataset_path),
        validation_seeds=(7003,),
    )
    hybrid = run_experiment(hybrid_config)
    hybrid_training = json.loads(
        (
            Path(hybrid.artifact_dir) / "data" / "training.json"
        ).read_text(encoding="utf-8")
    )
    assert hybrid.validation["split"] == "validation"
    assert hybrid.artifact_check["ok"]
    assert hybrid_training["training_config_hash"] == hybrid.config_hash
    resumed_hybrid = run_experiment(
        replace(
            hybrid_config,
            total_transitions=2,
            resume_checkpoint=hybrid.policy_path,
            output={
                **dict(hybrid_config.output),
                "name": "rlpd-e2e-resume",
            },
        )
    )
    assert resumed_hybrid.backend["final_step"] == 2
    resumed_hybrid_training = json.loads(
        (
            Path(resumed_hybrid.artifact_dir)
            / "data"
            / "training.json"
        ).read_text(encoding="utf-8")
    )
    assert resumed_hybrid_training["starting_step"] == 1


def test_sac_records_multiple_training_specs_and_valid_artifacts(tmp_path):
    config = RLTrainingConfig(
        track_id=TRACK_ID,
        algorithm_id="sac",
        training_seed=32,
        total_transitions=4,
        n_envs=2,
        algorithm={
            "batch_size": 2,
            "torch_threads": 1,
            "utd_ratio": 1.0,
            "vector_backend": "dummy",
            "verbose": 0,
        },
        replay={"capacity": 32, "learning_starts": 1},
        evaluation={"every_transitions": 0},
        output={
            "directory": str(tmp_path / "sac"),
            "name": "sac-e2e",
            "save_rollout": False,
        },
        validation_seeds=(7002,),
    )
    result = run_experiment(config)
    artifact_dir = Path(result.artifact_dir)
    training = json.loads(
        (artifact_dir / "data" / "training.json").read_text(
            encoding="utf-8"
        )
    )
    specs = training["training_episode_specs"]

    assert result.validation["split"] == "validation"
    assert result.artifact_check["ok"]
    assert training["training_config_hash"] == result.config_hash
    assert training["unique_training_episode_specs"] >= 2
    assert len({row["episode_spec_hash"] for row in specs}) >= 2
    assert {row["distribution_id"] for row in specs} == {
        "quadruple-regulation-training-l2-v1"
    }
    assert check_benchmark_artifacts(artifact_dir)["ok"]
    resumed = run_experiment(
        replace(
            config,
            total_transitions=6,
            resume_checkpoint=result.policy_path,
            output={
                **dict(config.output),
                "name": "sac-e2e-resume",
            },
        )
    )
    assert resumed.backend["final_step"] == 6
    resumed_training = json.loads(
        (
            Path(resumed.artifact_dir) / "data" / "training.json"
        ).read_text(encoding="utf-8")
    )
    assert resumed_training["starting_step"] == 4

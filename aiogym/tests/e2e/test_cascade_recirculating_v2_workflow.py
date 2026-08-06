from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.e2e

from aiogym import load_track, make_controller, make_env
from aiogym.benchmarks import evaluate_policy_on_track
from aiogym.cli.benchmark import main as benchmark_main
from aiogym.cli.evaluate import main as evaluate_main
from aiogym.controllers.checkpoints import (
    checkpoint_sha256,
    learned_policy_spec_for_track,
    load_policy_checkpoint,
)
from aiogym.datasets.collect import collect_dataset
from aiogym.datasets.config import DatasetCollectionConfig
from aiogym.datasets.reader import DatasetReader, validate_dataset
from aiogym.evaluation.artifact import check_benchmark_artifacts
from aiogym.models.registry import make_model
from aiogym.rl import RLTrainingConfig, run_experiment


TRACK_ID = "cascade-recirculating-regulation-generalist-v2"
DATASET_ID = "cascade-recirculating-e2e-v2"
DISTRIBUTION_ID = "cascade-recirculating-regulation-training-l2-v2"


def test_collect_train_reload_validate_and_benchmark(tmp_path):
    env = make_env(
        "cascade-recirculating",
        case="commissioning",
        reward_spec="regulation-v1",
    )
    try:
        observation, _ = env.reset(seed=41)
        assert env.action_space.shape == (4,)
        next_observation, reward, terminated, truncated, _ = env.step(
            np.full(env.action_space.shape, 0.5, dtype=np.float32)
        )
        assert env.observation_space.contains(observation)
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()

    dataset_path = tmp_path / "dataset"
    collection = DatasetCollectionConfig(
        {
            "schema_version": "aiogym.dataset_collection.v1",
            "track_id": TRACK_ID,
            "dataset_id": DATASET_ID,
            "split": "training",
            "base_seed": 41,
            "target_transitions": 1,
            "workers": 1,
            "collectors": [{"id": "nominal_pid", "weight": 1.0}],
            "output": str(dataset_path),
        }
    )
    collected = collect_dataset(collection)
    integrity = validate_dataset(dataset_path)
    reader = DatasetReader(dataset_path, verify_checksums=True)
    records = reader.metadata_records()
    episode = reader.load_episode(0)

    assert collected["actual_transitions"] == 1600
    assert integrity["ok"]
    assert reader.manifest["dataset_id"] == DATASET_ID
    assert len(records) == 1
    metadata = episode.metadata
    assert metadata["split"] == "training"
    assert metadata["track_id"] == TRACK_ID
    assert metadata["scenario"] == "cascade-recirculating"
    assert metadata["distribution_id"] == DISTRIBUTION_ID
    assert metadata["reward_spec_id"] == "regulation-v1"
    assert metadata["sensor_model"]["kind"] == "sensor_dynamics_v1"
    assert metadata["actuator_model"]["kind"] == "actuator_dynamics_v1"
    assert episode.array("action_policy_normalized").shape == (1600, 4)
    assert episode.array("action_commanded_physical").shape == (1600, 4)
    assert episode.array("action_applied_physical").shape == (1600, 4)

    common = {
        "track_id": TRACK_ID,
        "training_seed": 42,
        "n_envs": 1,
        "device": "cpu",
        "validation_seeds": (7042,),
    }
    configurations = (
        RLTrainingConfig(
            **common,
            algorithm_id="bc",
            total_transitions=1,
            algorithm={"batch_size": 2, "hidden": 8},
            output={
                "directory": str(tmp_path / "bc"),
                "name": "cascade-recirculating-bc",
            },
            dataset_id=DATASET_ID,
            dataset_path=str(dataset_path),
        ),
        RLTrainingConfig(
            **common,
            algorithm_id="sac",
            total_transitions=2,
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
                "directory": str(tmp_path / "sac"),
                "name": "cascade-recirculating-sac",
                "save_rollout": False,
            },
        ),
        RLTrainingConfig(
            **common,
            algorithm_id="rlpd",
            total_transitions=1,
            algorithm={
                "batch_size": 2,
                "bc_steps": 0,
                "n_critics": 2,
                "offline_fraction": 0.5,
                "pretrain_updates": 1,
                "utd_ratio": 1.0,
            },
            replay={"capacity": 16},
            evaluation={"every_transitions": 100},
            output={
                "directory": str(tmp_path / "rlpd"),
                "name": "cascade-recirculating-rlpd",
                "save_rollout": False,
            },
            dataset_id=DATASET_ID,
            dataset_path=str(dataset_path),
        ),
    )

    track = load_track(TRACK_ID)
    trained = {}
    run_results = {}
    for config in configurations:
        result = run_experiment(config)
        digest = checkpoint_sha256(result.policy_path)
        controller = load_policy_checkpoint(
            learned_policy_spec_for_track(
                result.policy_path,
                config.algorithm_id,
                digest,
                track,
            ),
            device="cpu",
        )
        assert result.validation["split"] == "validation"
        assert result.validation["aggregate"]["case_count"] == 2
        assert result.artifact_check["ok"]
        assert check_benchmark_artifacts(result.artifact_dir)["ok"]
        assert Path(result.policy_path).is_file()
        checkpoint = controller.metadata()["checkpoint"]
        assert checkpoint["sha256"] == digest
        assert controller.metadata()["normalized_actions"] is True
        assert controller.metadata()["action_mode"] == "actuator"
        trained[config.algorithm_id] = controller
        run_results[config.algorithm_id] = result

    model = make_model(track.scenario)
    controllers = {
        "pid": make_controller("pid", model=model, scenario=track.scenario),
        "bc": trained["bc"],
    }
    for controller in controllers.values():
        benchmark = evaluate_policy_on_track(
            controller,
            track,
            base_seeds=(7043,),
        )
        assert benchmark["split"] == "validation"
        assert benchmark["aggregate"]["case_count"] == 2
        assert benchmark["aggregate"]["ranking_eligible"] in {True, False}

    bc_result = run_results["bc"]
    bc_digest = checkpoint_sha256(bc_result.policy_path)
    evaluate_output = tmp_path / "evaluate.json"
    assert evaluate_main(
        [
            "--checkpoint",
            bc_result.policy_path,
            "--track",
            TRACK_ID,
            "--algorithm",
            "bc",
            "--sha256",
            bc_digest,
            "--seeds",
            "7042",
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
            bc_result.policy_path,
            "--algorithm",
            "bc",
            "--sha256",
            bc_digest,
            "--name",
            "bc",
            "--seeds",
            "7042",
            "--output",
            str(benchmark_output),
        ]
    ) == 0
    evaluated = json.loads(evaluate_output.read_text(encoding="utf-8"))
    benchmarked = json.loads(
        benchmark_output.read_text(encoding="utf-8")
    )["results"]["bc"]
    assert evaluated["split"] == "validation"
    assert evaluated["aggregate"] == bc_result.validation["aggregate"]
    assert benchmarked["aggregate"] == bc_result.validation["aggregate"]

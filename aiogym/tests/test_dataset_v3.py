from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import aiogym.scenarios  # noqa: F401
from aiogym.controllers.base import make_controller
from aiogym.core import make_env, rollout
from aiogym.workflows import DatasetReader, collect, load_plant


DESIGN = (
    Path(__file__).resolve().parents[2]
    / "configs/design/cascade-recirculating-example-v1.json"
)


def test_three_tank_collect_round_trip_matches_core_rollout(tmp_path):
    plant = load_plant(DESIGN)
    output = tmp_path / "three-tank-pid"
    result = collect(
        task="three_tank/regulation",
        plant=plant,
        policy="pid",
        episodes=2,
        seed=5,
        output=output,
        max_steps=3,
    )
    reader = DatasetReader(output, verify_checksums=True)
    assert result["episodes"] == len(reader) == 2
    assert result["transitions"] == reader.transition_count == 6
    assert reader.manifest["schema_version"] == "aiogym.dataset.v3"
    assert reader.manifest["plant_hash"] == plant.plant_hash
    assert "track_id" not in reader.manifest
    assert "distribution_id" not in reader.manifest
    assert sorted(path.name for path in output.glob("episode-*.npz")) == [
        "episode-000000.npz",
        "episode-000001.npz",
    ]

    env = make_env("three_tank/regulation", plant=plant)
    try:
        expected = rollout(
            env,
            make_controller("pid", env=env),
            seed=5,
            max_steps=3,
        )
    finally:
        env.close()
    episode = reader.load_episode(0)
    assert np.array_equal(
        episode.array("observation"),
        np.asarray([row.observation for row in expected.transitions]),
    )
    assert np.array_equal(
        episode.array("action"),
        np.asarray([row.action for row in expected.transitions]),
    )
    assert np.array_equal(
        episode.array("next_observation"),
        np.asarray([row.next_observation for row in expected.transitions]),
    )


def test_resume_appends_only_missing_episodes_and_no_overwrite(tmp_path):
    output = tmp_path / "resume"
    first = collect(
        task="quadruple/regulation",
        preset="minimum-phase",
        policy="random",
        episodes=1,
        seed=17,
        output=output,
        max_steps=2,
    )
    first_hash = first["manifest"]["episodes"][0]["content_hash"]
    with pytest.raises(FileExistsError):
        collect(
            task="quadruple/regulation",
            preset="minimum-phase",
            policy="random",
            episodes=2,
            seed=17,
            output=output,
            max_steps=2,
        )
    resumed = collect(
        task="quadruple/regulation",
        preset="minimum-phase",
        policy="random",
        episodes=2,
        seed=17,
        output=output,
        max_steps=2,
        resume=True,
    )
    assert resumed["episodes"] == 2
    assert resumed["manifest"]["episodes"][0]["content_hash"] == first_hash
    repeated = collect(
        task="quadruple/regulation",
        preset="minimum-phase",
        policy="random",
        episodes=2,
        seed=17,
        output=output,
        max_steps=2,
        resume=True,
    )
    assert repeated["manifest"]["episodes"] == resumed["manifest"]["episodes"]
    with pytest.raises(ValueError, match="identity"):
        collect(
            task="quadruple/regulation",
            preset="minimum-phase",
            policy="random",
            episodes=3,
            seed=18,
            output=output,
            max_steps=2,
            resume=True,
        )


def test_pid_mpc_random_and_checkpoint_compatible_policy_objects_collect(tmp_path):
    for controller_id in ("pid", "mpc", "random"):
        result = collect(
            task="quadruple/regulation",
            preset="minimum-phase",
            policy=controller_id,
            episodes=1,
            seed=2,
            output=tmp_path / controller_id,
            max_steps=2,
        )
        assert result["transitions"] == 2

    class PredictPolicy:
        def reset(self, seed=None):
            self.seed = seed

        def act(self, observation, context):
            del observation, context
            return np.asarray([0.3, 0.3], dtype=np.float32)

        def metadata(self):
            return {"id": "checkpoint-like", "algorithm": "sac"}

    learned = collect(
        task="quadruple/regulation",
        preset="minimum-phase",
        policy=PredictPolicy(),
        episodes=1,
        seed=2,
        output=tmp_path / "learned",
        max_steps=2,
    )
    assert learned["manifest"]["policy"]["algorithm"] == "sac"


def test_v2_migration_is_one_shot_and_preserves_legacy_metadata(tmp_path):
    from aiogym.datasets.collect import collect_dataset as collect_v2
    from aiogym.datasets.config import DatasetCollectionConfig
    from aiogym.datasets.reader import DatasetReader as DatasetV2Reader
    from scripts.migrate_dataset_v2_to_v3 import migrate

    source = tmp_path / "v2"
    collect_v2(
        DatasetCollectionConfig(
            {
                "schema_version": "aiogym.dataset_collection.v1",
                "track_id": "quadruple-regulation-generalist-v1",
                "dataset_id": "migration-sample-v2",
                "split": "training",
                "base_seed": 23,
                "target_transitions": 1,
                "workers": 1,
                "collectors": [{"id": "nominal_pid", "weight": 1.0}],
                "output": str(source),
            }
        )
    )
    output = tmp_path / "v3"
    report = migrate(source, output)
    assert report["ok"]
    old = DatasetV2Reader(source).load_episode(0)
    new_reader = DatasetReader(output, verify_checksums=True)
    new = new_reader.load_episode(0)
    assert np.allclose(
        new.array("action"),
        0.5 * (old.array("action_policy_normalized") + 1.0),
    )
    assert np.array_equal(new.array("reward"), old.array("reward_scalar"))
    assert new_reader.manifest["legacy_metadata"]["source_schema"] == "aiogym.dataset.v2"
    assert new.metadata["legacy_metadata"]["track_id"] == old.metadata["track_id"]


def test_v3_runtime_reader_has_no_v2_compatibility_imports():
    source = (
        Path(__file__).resolve().parents[1] / "workflows/dataset.py"
    ).read_text(encoding="utf-8")
    assert "aiogym.datasets" not in source
    assert "dataset.v2" not in source

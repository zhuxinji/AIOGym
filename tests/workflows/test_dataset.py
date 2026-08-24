from __future__ import annotations

import json
import numpy as np
import pytest

from aiogym import make_controller, make_env
from aiogym.core.rollout import rollout
from aiogym.workflows import DatasetReader, collect
from aiogym.workflows.dataset import DatasetWriter, REQUIRED_ARRAYS


def test_collect_uses_the_given_env_and_round_trips_episodes(tmp_path):
    output = tmp_path / "three-tank-pid"
    env = make_env("three_tank", reward="regulation")
    try:
        policy = make_controller("pid", env=env)
        result = collect(
            env=env,
            policy=policy,
            episodes=2,
            seed=5,
            output=output,
            max_steps=3,
        )
        expected = rollout(env, policy, seed=5, max_steps=3)
        observation, _ = env.reset(seed=99)
        assert observation.shape == env.observation_space.shape
    finally:
        env.close()

    reader = DatasetReader(output)
    assert result["episodes"] == len(reader) == 2
    assert result["transitions"] == reader.transition_count == 6
    assert reader.metadata["schema_version"] == "aiogym.dataset.v2"
    assert reader.metadata["environment"]["scenario"] == "three_tank"
    assert reader.metadata["environment"]["benchmark"] is None
    assert reader.metadata["environment"]["randomize"] is False
    assert reader.metadata["environment"]["disturbance"] is None
    assert reader.metadata["environment"]["reward"] == "regulation"
    assert reader.metadata["environment"]["parameters"]["pump_power_max"] == 370.0
    assert sorted(path.name for path in output.glob("episode-*.npz")) == [
        "episode-000000.npz",
        "episode-000001.npz",
    ]
    episode = reader[0]
    assert episode.episode_id == "episode-000000"
    assert episode.transition_count == 3
    assert np.array_equal(
        episode.array("observation"),
        np.asarray([row.observation for row in expected.transitions]),
    )
    assert np.array_equal(
        episode.array("action"),
        np.asarray([row.action for row in expected.transitions]),
    )
    assert [row.episode_id for row in reader.iter_episodes()] == [
        "episode-000000",
        "episode-000001",
    ]


def test_collect_rejects_an_existing_nonempty_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "operator-notes.txt").write_text("keep", encoding="utf-8")
    env = make_env("quadruple")
    try:
        with pytest.raises(FileExistsError, match="non-empty"):
            collect(env=env, output=output, max_steps=2)
    finally:
        env.close()
    assert (output / "operator-notes.txt").read_text(encoding="utf-8") == "keep"


def test_collect_accepts_builtin_and_policy_object(tmp_path):
    env = make_env("quadruple")
    try:
        for controller_id in ("pid", "mpc", "random"):
            result = collect(
                env=env,
                policy=controller_id,
                episodes=1,
                seed=2,
                output=tmp_path / controller_id,
                max_steps=2,
            )
            assert result["transitions"] == 2

        class Policy:
            env = None

            def reset(self, seed=None):
                self.seed = seed

            def act(self, observation, context):
                del observation, context
                return np.asarray([0.3, 0.3], dtype=np.float32)

            def metadata(self):
                return {"id": "learned", "algorithm": "sac"}

        learned = collect(
            env=env,
            policy=Policy(),
            episodes=1,
            seed=2,
            output=tmp_path / "learned",
            max_steps=2,
        )
    finally:
        env.close()
    assert learned["metadata"]["policy"] == {
        "id": "learned",
        "algorithm": "sac",
    }


def test_writer_rejects_missing_or_inconsistent_arrays(tmp_path):
    writer = DatasetWriter(
        tmp_path / "missing",
        environment={
            "scenario": "toy",
            "benchmark": None,
            "reward": "regulation",
            "parameters": {},
        },
        policy={"id": "random"},
        base_seed=0,
    )
    arrays = {name: np.zeros(2) for name in REQUIRED_ARRAYS}
    arrays["step_index"] = np.arange(2)
    arrays["physical_time"] = np.asarray([1.0, 2.0])
    missing = dict(arrays)
    del missing["action"]
    with pytest.raises(ValueError, match="missing"):
        writer.append(0, 0, missing)

    inconsistent = dict(arrays)
    inconsistent["action"] = np.zeros(3)
    with pytest.raises(ValueError, match="inconsistent length"):
        writer.append(0, 0, inconsistent)


def test_reader_rejects_unknown_schema_and_inconsistent_metadata(tmp_path):
    output = tmp_path / "dataset"
    env = make_env("quadruple")
    try:
        collect(env=env, output=output, max_steps=2)
    finally:
        env.close()
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "invalid"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported dataset schema"):
        DatasetReader(output)

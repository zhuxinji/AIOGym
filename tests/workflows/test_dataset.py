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
    assert reader.metadata["schema_version"] == "aiogym.dataset.v4"
    assert reader.metadata["environment"]["scenario"] == "three_tank"
    assert reader.metadata["environment"]["benchmark"] is None
    assert reader.metadata["environment"]["randomize"] is False
    assert reader.metadata["environment"]["boundary_probability"] == 0.0
    assert reader.metadata["environment"]["disturbance"] is None
    assert reader.metadata["environment"]["reward"] == "regulation"
    assert reader.metadata["environment"]["parameters"]["pump_power_max"] == 370.0
    assert reader.metadata["environment"]["initial_state"] is None
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
        "declared_environment_check": "not_provided",
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


@pytest.fixture
def small_dataset(tmp_path):
    env = make_env("heater")
    path = tmp_path / "dataset"
    try:
        collect(env=env, policy="hold", max_steps=2, output=path)
    finally:
        env.close()
    return path


@pytest.mark.parametrize("version", ["aiogym.dataset.v2", "aiogym.dataset.v3", "aiogym.dataset.v4"])
def test_training_dataset_interface_requirement_and_legacy_warning(small_dataset, version):
    from aiogym.rl.datasets import load_training_dataset
    from aiogym.workflows.dataset import _metadata_digest

    path = small_dataset / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = version
    del metadata["environment"]["policy_interface"]
    metadata["content_sha256"] = _metadata_digest(metadata)
    path.write_text(json.dumps(metadata))
    env = make_env("heater")
    try:
        if version == "aiogym.dataset.v4":
            with pytest.raises(ValueError, match="missing compatibility fields.*policy_interface"):
                load_training_dataset(small_dataset, env=env)
        else:
            with pytest.warns(UserWarning, match="only legacy compatibility checks"):
                reader = load_training_dataset(small_dataset, env=env)
            assert reader.transition_count == 2
    finally:
        env.close()


@pytest.mark.parametrize("field, value, message", [
    ("reward", np.zeros((2, 1)), "dimensions"),
    ("observation", np.full((2, 1), np.nan), "finite"),
    ("terminated", np.zeros(2, dtype=int), "boolean"),
    ("truncated", np.asarray([True, False]), "after termination or truncation"),
])
def test_legacy_reader_enforces_numeric_shapes_and_done_flags(small_dataset, field, value, message):
    metadata_path = small_dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = "aiogym.dataset.v2"
    metadata.pop("content_sha256")
    for row in metadata["episodes"]:
        row.pop("sha256")
    metadata_path.write_text(json.dumps(metadata))
    path = small_dataset / "episode-000000.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays[field] = value
    np.savez_compressed(path, **arrays)
    with pytest.raises((ValueError, TypeError), match=message):
        DatasetReader(small_dataset)[0]


def test_dataset_fingerprint_survives_move_and_repack_but_detects_tampering(small_dataset, tmp_path):
    import shutil

    original = DatasetReader(small_dataset).content_sha256
    moved = tmp_path / "moved"
    shutil.move(small_dataset, moved)
    path = moved / "episode-000000.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    np.savez(path, **dict(reversed(list(arrays.items()))))
    assert DatasetReader(moved).content_sha256 == original
    arrays["reward"][0] -= 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="episode content digest mismatch"):
        DatasetReader(moved)[0]
    metadata_path = moved / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["environment"]["control_dt"] *= 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata content digest mismatch"):
        DatasetReader(moved)


def test_training_dataset_caches_validated_snapshot_and_rejects_next_observation_bounds(
    small_dataset, monkeypatch,
):
    from aiogym.rl.datasets import load_training_dataset, training_dataset_metadata
    from aiogym.workflows.dataset import _arrays_digest, _metadata_digest

    env = make_env("heater")
    try:
        reader = load_training_dataset(small_dataset, env=env)
        expected = reader.content_sha256
        with monkeypatch.context() as patch:
            def unexpected_reload(*args, **kwargs):
                raise AssertionError("training snapshot should not reopen episodes")
            patch.setattr(np, "load", unexpected_reload)
            assert training_dataset_metadata(reader)["content_sha256"] == expected
            assert tuple(reader.iter_episodes())[0] is reader[0]
        path = small_dataset / "episode-000000.npz"
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays["next_observation"][0, 0] = -1e6
        np.savez_compressed(path, **arrays)
        metadata_path = small_dataset / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["episodes"][0]["sha256"] = _arrays_digest(arrays)
        metadata["content_sha256"] = _metadata_digest(metadata)
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="next_observation must belong"):
            load_training_dataset(small_dataset, env=env)
    finally:
        env.close()


def test_training_dataset_records_different_source_variation(tmp_path):
    from aiogym.rl.datasets import load_training_dataset, training_dataset_metadata

    source = make_env("heater", randomize=True, noise=True)
    target = make_env("heater")
    try:
        path = tmp_path / "noisy"
        collect(env=source, policy="hold", max_steps=2, output=path)
        reader = load_training_dataset(path, env=target)
        provenance = training_dataset_metadata(reader)
        assert provenance["environment"]["noise"] == reader.metadata["environment"]["noise"]
        assert provenance["environment"]["noise"] is not None
        assert target.unwrapped.runtime_config["noise"] is None
    finally:
        source.close()
        target.close()

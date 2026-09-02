from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import aiogym
from aiogym.rl.algorithms import get_algorithm
from aiogym.rl.datasets import load_training_dataset
from aiogym.workflows._checkpoint import load_training_checkpoint


pytest.importorskip("torch")
pytestmark = pytest.mark.rl

SMALL_RLPD = {
    "batch_size": 4,
    "buffer_size": 32,
    "critic_subset": 1,
    "device": "cpu",
    "hidden_sizes": [8, 8],
    "learning_starts": 1,
    "n_critics": 2,
    "utd_ratio": 1,
}


def test_rlpd_requires_dataset_before_creating_output(tmp_path):
    env = aiogym.make_env("quadruple")
    output = tmp_path / "training"
    try:
        with pytest.raises(ValueError, match="rlpd requires dataset"):
            aiogym.train(
                env=env,
                algorithm="rlpd",
                steps=1,
                algorithm_kwargs=SMALL_RLPD,
                output=output,
            )
    finally:
        env.close()
    assert not output.exists()


@pytest.mark.parametrize(
    ("batch_size", "offline_ratio", "offline_count", "online_count"),
    [
        (4, 0.0, 0, 4),
        (4, 0.25, 1, 3),
        (4, 0.5, 2, 2),
        (4, 1.0, 4, 0),
        (3, 0.5, 1, 2),
    ],
)
def test_rlpd_samples_configured_offline_ratio(
    tmp_path, batch_size, offline_ratio, offline_count, online_count
):
    env = aiogym.make_env("quadruple")
    dataset_path = tmp_path / "dataset"
    try:
        aiogym.collect(
            env=env,
            policy="hold",
            max_steps=4,
            output=dataset_path,
        )
        reader = load_training_dataset(dataset_path, env=env)
        backend = get_algorithm("rlpd")
        kwargs = backend.effective_kwargs(
            steps=1,
            values={
                **SMALL_RLPD,
                "batch_size": batch_size,
                "offline_ratio": offline_ratio,
            },
        )
        model = backend.create(env=env, seed=3, algorithm_kwargs=kwargs)
        model.load_offline_dataset(reader)
        observation, _ = env.reset(seed=3)
        action = model.random_action()
        next_observation, reward, terminated, _truncated, _info = env.step(action)
        model.add_online_transition(
            observation,
            action,
            reward,
            next_observation,
            terminated,
        )
        batch = model.sample_batch()
    finally:
        env.close()
    assert batch["observation"].shape == (batch_size, 6)
    assert batch["action"].shape == (batch_size, 2)
    assert model.offline_samples == offline_count
    assert model.online_samples == online_count


def test_rlpd_defaults_match_reference_runner():
    backend = get_algorithm("rlpd")
    kwargs = backend.effective_kwargs(steps=1, values={})
    assert kwargs["learning_starts"] == 10_000
    assert kwargs["offline_ratio"] == 0.5


def test_rlpd_uses_complete_workflow_and_standard_checkpoint(tmp_path):
    env = aiogym.make_env("quadruple")
    dataset_path = tmp_path / "dataset"
    training_path = tmp_path / "training"
    comparison_path = tmp_path / "comparison"
    try:
        aiogym.collect(
            env=env,
            policy="pid",
            max_steps=4,
            output=dataset_path,
        )
        result = aiogym.train(
            env=env,
            algorithm="rlpd",
            steps=2,
            seed=7,
            algorithm_kwargs=SMALL_RLPD,
            dataset=dataset_path,
            record_every=1,
            output=training_path,
        )
        policy = aiogym.load_policy(result["checkpoint"], env=env)
        observation, _ = env.reset(seed=9)
        action = policy.act(observation, {})
        evaluation = aiogym.evaluate(
            env=env,
            policy=policy,
            seeds=[9],
            max_steps=2,
        )
        comparison = aiogym.compare_policies(
            env=env,
            policies={"rlpd": policy, "hold": "hold"},
            seeds=[9],
            max_steps=2,
            output=comparison_path,
        )
    finally:
        env.close()

    metadata = json.loads(
        (training_path / "metadata.json").read_text(encoding="utf-8")
    )
    with zipfile.ZipFile(training_path / "model.zip") as checkpoint:
        manifest = json.loads(checkpoint.read("manifest.json"))
    assert result["schema_version"] == "aiogym.training.v12"
    assert result["algorithm"] == "rlpd"
    assert result["dataset"]["transition_count"] == 4
    assert result["behavior_cloning"] is None
    assert metadata["dataset"]["path"] == str(dataset_path.resolve())
    assert manifest["algorithm"] == "rlpd"
    assert manifest["runtime"]["paper"] == "https://arxiv.org/abs/2302.02948"
    assert env.action_space.contains(action)
    assert np.isfinite(action).all()
    assert evaluation["policy"]["algorithm"] == "rlpd"
    assert set(comparison["evaluations"]) == {"hold", "rlpd"}


def test_rlpd_training_continues_full_training_state(tmp_path):
    env = aiogym.make_env("quadruple")
    dataset_path = tmp_path / "dataset"
    try:
        aiogym.collect(
            env=env,
            policy="hold",
            max_steps=4,
            output=dataset_path,
        )
        first = aiogym.train(
            env=env,
            algorithm="rlpd",
            steps=2,
            seed=5,
            algorithm_kwargs=SMALL_RLPD,
            dataset=dataset_path,
            output=tmp_path / "first",
        )
        _, _, first_model, _ = load_training_checkpoint(
            first["checkpoint"], env=env, algorithm="rlpd"
        )
        first_updates = first_model.gradient_updates
        first_replay_size = len(first_model.online)

        continued = aiogym.train(
            env=env,
            algorithm="rlpd",
            steps=2,
            dataset=dataset_path,
            resume_from=first["checkpoint"],
            output=tmp_path / "continued",
        )
        _, _, continued_model, state = load_training_checkpoint(
            continued["checkpoint"], env=env, algorithm="rlpd"
        )
    finally:
        env.close()

    assert continued["initial_steps"] == 2
    assert continued["added_steps"] == 2
    assert continued["actual_steps"] == 4
    assert state["completed_steps"] == 4
    assert continued_model.environment_steps == 4
    assert len(continued_model.online) == first_replay_size + 2
    assert continued_model.gradient_updates > first_updates


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"n_critics": 1}, "n_critics must be at least 2"),
        ({"n_critics": 2, "critic_subset": 3}, "must not exceed"),
        ({"utd_ratio": 0}, "utd_ratio must be a positive integer"),
        ({"offline_ratio": -0.1}, "offline_ratio must be in"),
        ({"offline_ratio": 1.1}, "offline_ratio must be in"),
        ({"offline_ratio": "0.5"}, "offline_ratio must be a number"),
    ],
)
def test_rlpd_validates_canonical_configuration(values, error):
    backend = get_algorithm("rlpd")
    with pytest.raises((TypeError, ValueError), match=error):
        backend.effective_kwargs(steps=1, values=values)

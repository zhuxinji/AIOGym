from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

pytestmark = pytest.mark.rl

from aiogym import collect, load_policy, make_controller, make_env, train


SMALL_POLICY = {"policy_kwargs": {"net_arch": [8, 8]}}


def test_behavior_cloning_pretrains_and_saves_the_standard_sb3_checkpoint(tmp_path):
    env = make_env("quadruple", reward="regulation")
    dataset = tmp_path / "dataset"
    output = tmp_path / "training"
    try:
        expert = make_controller(
            "hold", env=env, config={"action": [0.2, 0.8]}
        )
        collect(
            env=env,
            policy=expert,
            episodes=2,
            max_steps=10,
            output=dataset,
        )
        result = train(
            env=env,
            algorithm="sac",
            steps=1,
            seed=0,
            algorithm_kwargs=SMALL_POLICY,
            dataset=dataset,
            behavior_cloning_epochs=30,
            behavior_cloning_batch_size=8,
            behavior_cloning_learning_rate=0.01,
            output=output,
        )
        policy = load_policy(result["checkpoint"], env=env)
        observation, _ = env.reset(seed=0)
        action = policy.act(observation, {})
    finally:
        env.close()

    report = json.loads(
        (output / "behavior_cloning.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "aiogym.behavior_cloning.v1"
    assert report["action_field"] == "commanded_action"
    assert report["transition_count"] == 20
    assert report["loss"][-1]["mean_loss"] < report["loss"][0]["mean_loss"]
    assert metadata["behavior_cloning"]["artifact"] == "behavior_cloning.json"
    assert result["behavior_cloning_artifact"] == str(
        (output / "behavior_cloning.json").resolve()
    )
    assert (output / "model.zip").is_file()
    assert action == pytest.approx([0.2, 0.8], abs=0.02)


@pytest.mark.parametrize("algorithm", ("ddpg", "td3"))
def test_behavior_cloning_supports_the_other_off_policy_actors(
    tmp_path, algorithm
):
    env = make_env("quadruple", reward="regulation")
    dataset = tmp_path / f"{algorithm}-dataset"
    output = tmp_path / f"{algorithm}-training"
    try:
        collect(env=env, policy="hold", max_steps=2, output=dataset)
        result = train(
            env=env,
            algorithm=algorithm,
            steps=1,
            algorithm_kwargs=SMALL_POLICY,
            dataset=dataset,
            behavior_cloning_epochs=1,
            behavior_cloning_batch_size=2,
            output=output,
        )
        policy = load_policy(result["checkpoint"], env=env)
        observation, _ = env.reset(seed=0)
        action = policy.act(observation, {})
    finally:
        env.close()
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    report = json.loads(
        (output / "behavior_cloning.json").read_text(encoding="utf-8")
    )
    assert report["algorithm"] == algorithm


def test_behavior_cloning_rejects_incompatible_scenario_before_writing_output(
    tmp_path,
):
    source_env = make_env("quadruple", reward="regulation")
    dataset = tmp_path / "dataset"
    try:
        collect(env=source_env, episodes=1, max_steps=2, output=dataset)
    finally:
        source_env.close()
    target_env = make_env("three_tank", reward="regulation")
    output = tmp_path / "training"
    try:
        with pytest.raises(ValueError, match="scenario does not match"):
            train(
                env=target_env,
                algorithm="sac",
                steps=1,
                dataset=dataset,
                behavior_cloning_epochs=1,
                output=output,
            )
    finally:
        target_env.close()
    assert not output.exists()
def test_behavior_cloning_requires_an_off_policy_actor_and_explicit_epochs(tmp_path):
    env = make_env("quadruple")
    try:
        with pytest.raises(ValueError, match="does not consume a training Dataset"):
            train(
                env=env,
                algorithm="ppo",
                steps=1,
                dataset=tmp_path / "dataset",
                behavior_cloning_epochs=1,
                output=tmp_path / "ppo",
            )
        with pytest.raises(ValueError, match="requires behavior_cloning_epochs"):
            train(
                env=env,
                algorithm="sac",
                steps=1,
                dataset=tmp_path / "dataset",
                output=tmp_path / "missing-epochs",
            )
        with pytest.raises(ValueError, match="requires dataset"):
            train(
                env=env,
                algorithm="sac",
                steps=1,
                behavior_cloning_epochs=1,
                output=tmp_path / "missing-dataset",
            )
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            train(
                env=env,
                algorithm="sac",
                steps=1,
                dataset=tmp_path / "dataset",
                behavior_cloning_epochs=1,
                behavior_cloning_batch_size=0,
                output=tmp_path / "invalid-batch",
            )
        with pytest.raises(ValueError, match="learning_rate must be a positive"):
            train(
                env=env,
                algorithm="sac",
                steps=1,
                dataset=tmp_path / "dataset",
                behavior_cloning_epochs=1,
                behavior_cloning_learning_rate=float("nan"),
                output=tmp_path / "invalid-learning-rate",
            )
    finally:
        env.close()

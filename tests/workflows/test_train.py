from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

pytestmark = pytest.mark.rl

from aiogym import load_policy, make_env, plot_training_curve, train
from aiogym.workflows._sb3_runtime import (
    algorithm_class,
    effective_algorithm_kwargs,
)


SMALL_POLICY = {"policy_kwargs": {"net_arch": [8, 8]}}
QUADRUPLE_SAC_PROFILE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "quadruple-sac.json"
    ).read_text(encoding="utf-8")
)
QUADRUPLE_PPO_PROFILE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "quadruple-ppo.json"
    ).read_text(encoding="utf-8")
)
QUADRUPLE_DDPG_PROFILE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "quadruple-ddpg.json"
    ).read_text(encoding="utf-8")
)
THREE_TANK_SAC_PROFILE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "three-tank-residual-sac.json"
    ).read_text(encoding="utf-8")
)


def test_off_policy_training_saves_loads_and_predicts(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "sac"
    try:
        result = train(
            env=env,
            algorithm="sac",
            steps=2,
            seed=4,
            algorithm_kwargs=SMALL_POLICY,
            output=output,
        )
        policy = load_policy(
            output / "model.zip",
            algorithm="sac",
            env=env,
        )
        observation, _ = env.reset(seed=11)
        action = policy.act(observation, {})
    finally:
        env.close()

    assert result["schema_version"] == "aiogym.training.v4"
    assert result["algorithm"] == "sac"
    assert result["steps"] == 2
    assert result["actual_steps"] == 2
    assert result["seed"] == 4
    assert result["record_every"] == 500
    assert set(path.name for path in output.iterdir()) == {
        "metadata.json",
        "model.zip",
        "training_curve.json",
        "training_curve.svg",
    }
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["algorithm"] == "sac"
    assert metadata["steps"] == 2
    assert metadata["actual_steps"] == 2
    assert metadata["seed"] == 4
    assert metadata["algorithm_kwargs"] == effective_algorithm_kwargs(
        "sac", 2, SMALL_POLICY
    )
    assert metadata["environment"]["observation_shape"] == [8]
    assert metadata["environment"]["action_shape"] == [2]
    curve = json.loads(
        (output / "training_curve.json").read_text(encoding="utf-8")
    )
    assert curve["schema_version"] == "aiogym.training_curve.v1"
    assert [row["end_step"] for row in curve["records"]] == [2]
    assert [row["transition_count"] for row in curve["records"]] == [2]
    ET.parse(output / "training_curve.svg")


def test_quadruple_sac_profile_matches_validated_training_configuration():
    assert QUADRUPLE_SAC_PROFILE == {
        "batch_size": 256,
        "ent_coef": 0.0005,
        "gamma": 0.995,
        "learning_rate": 0.001,
        "learning_starts": 0,
    }
    resolved = effective_algorithm_kwargs(
        "sac", 50_000, QUADRUPLE_SAC_PROFILE
    )
    assert resolved["buffer_size"] == 50_001
    assert resolved["train_freq"] == 1
    assert resolved["gradient_steps"] == 1


def test_quadruple_ppo_profile_matches_validated_training_configuration():
    assert QUADRUPLE_PPO_PROFILE == {
        "batch_size": 100,
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
        "gamma": 0.995,
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "n_steps": 600,
    }
    resolved = effective_algorithm_kwargs(
        "ppo", 50_000, QUADRUPLE_PPO_PROFILE
    )
    assert resolved["policy"] == "MlpPolicy"


def test_quadruple_ddpg_profile_has_explicit_exploration_noise():
    assert QUADRUPLE_DDPG_PROFILE == {
        "action_noise": {"std": 0.1, "type": "normal"},
        "batch_size": 256,
        "gamma": 0.995,
        "learning_rate": 0.001,
        "learning_starts": 1000,
    }


def test_ddpg_training_materializes_json_action_noise(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "ddpg-noise"
    kwargs = {
        "action_noise": {"std": 0.1, "type": "normal"},
        "learning_starts": 2,
        **SMALL_POLICY,
    }
    try:
        result = train(
            env=env,
            algorithm="ddpg",
            steps=2,
            seed=0,
            algorithm_kwargs=kwargs,
            output=output,
        )
    finally:
        env.close()
    assert result["algorithm_kwargs"]["action_noise"] == {
        "std": 0.1,
        "type": "normal",
    }


def test_ddpg_training_accepts_ornstein_uhlenbeck_noise(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "ddpg-ou-noise"
    try:
        result = train(
            env=env,
            algorithm="ddpg",
            steps=2,
            seed=0,
            algorithm_kwargs={
                "action_noise": {
                    "std": 0.1,
                    "type": "ornstein-uhlenbeck",
                },
                "learning_starts": 2,
                **SMALL_POLICY,
            },
            output=output,
        )
    finally:
        env.close()
    assert result["algorithm_kwargs"]["action_noise"]["type"] == (
        "ornstein-uhlenbeck"
    )


def test_three_tank_sac_profile_matches_experimental_training_configuration():
    assert THREE_TANK_SAC_PROFILE == {
        "batch_size": 256,
        "ent_coef": 0.0001,
        "gamma": 0.9995,
        "learning_rate": 0.001,
        "learning_starts": 0,
    }


def test_training_curve_records_fixed_windows_and_can_be_replotted(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "sac"
    try:
        train(
            env=env,
            algorithm="sac",
            steps=5,
            seed=0,
            algorithm_kwargs=SMALL_POLICY,
            record_every=2,
            output=output,
        )
    finally:
        env.close()

    curve_path = output / "training_curve.json"
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    assert [row["start_step"] for row in curve["records"]] == [0, 2, 4]
    assert [row["end_step"] for row in curve["records"]] == [2, 4, 5]
    assert [row["transition_count"] for row in curve["records"]] == [2, 2, 1]
    assert all(np.isfinite(row["mean_reward"]) for row in curve["records"])

    replotted = plot_training_curve(
        curve_path,
        output=tmp_path / "replotted.svg",
    )
    root = ET.parse(replotted).getroot()
    text = " ".join(element.text or "" for element in root.iter())
    assert "environment steps" in text
    assert "0" in text


def test_on_policy_training_completes(tmp_path):
    env = make_env("quadruple")
    try:
        result = train(
            env=env,
            algorithm="ppo",
            steps=2,
            seed=5,
            algorithm_kwargs=SMALL_POLICY,
            output=tmp_path / "ppo",
        )
    finally:
        env.close()
    assert result["algorithm"] == "ppo"
    assert result["steps"] == 2


@pytest.mark.parametrize("algorithm", ("ddpg", "ppo", "sac", "td3"))
def test_supported_algorithm_constructors(algorithm):
    env = make_env("quadruple")
    try:
        kwargs = effective_algorithm_kwargs(algorithm, 2, SMALL_POLICY)
        policy = kwargs.pop("policy")
        model = algorithm_class(algorithm)(policy, env, seed=0, **kwargs)
        assert model.action_space.shape == env.action_space.shape
    finally:
        env.close()


def test_training_rejects_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    env = make_env("quadruple")
    try:
        with pytest.raises(FileExistsError, match="non-empty"):
            train(env=env, algorithm="sac", steps=1, output=output)
    finally:
        env.close()
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_training_rejects_benchmark_environment(tmp_path):
    env = make_env("quadruple", benchmark="tracking")
    try:
        with pytest.raises(ValueError, match="benchmark environment"):
            train(
                env=env,
                algorithm="sac",
                steps=1,
                output=tmp_path / "benchmark",
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"algorithm": "unknown", "steps": 1, "seed": 0}, ValueError),
        ({"algorithm": "sac", "steps": 0, "seed": 0}, ValueError),
        ({"algorithm": "sac", "steps": True, "seed": 0}, TypeError),
        ({"algorithm": "sac", "steps": 1, "seed": -1}, ValueError),
        (
            {"algorithm": "sac", "steps": 1, "seed": 0, "record_every": 0},
            ValueError,
        ),
        (
            {"algorithm": "sac", "steps": 1, "seed": 0, "record_every": True},
            TypeError,
        ),
    ],
)
def test_training_validates_inputs(tmp_path, kwargs, error):
    env = make_env("quadruple")
    try:
        with pytest.raises(error):
            train(env=env, output=tmp_path / "invalid", **kwargs)
    finally:
        env.close()

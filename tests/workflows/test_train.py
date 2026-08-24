from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

pytestmark = pytest.mark.rl

import aiogym
from aiogym import load_policy, make_env, plot_training_curve, train
from aiogym.rl import algorithms as algorithm_registry
from aiogym.rl.algorithms import get_algorithm
from aiogym.workflows._checkpoint import load_training_checkpoint
from aiogym.workflows.train import _is_better_training_evaluation


SMALL_POLICY = {"policy_kwargs": {"net_arch": [8, 8]}}
THREE_TANK_SAC_CONFIG = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "aiogym"
        / "rl"
        / "configs"
        / "three-tank-sac-nstep10.json"
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
            env=env,
        )
        observation, _ = env.reset(seed=11)
        action = policy.act(observation, {})
    finally:
        env.close()

    assert result["schema_version"] == "aiogym.training.v10"
    assert result["checkpoint_schema"] == "aiogym.checkpoint.v2"
    assert result["algorithm"] == "sac"
    assert result["steps"] == 2
    assert result["initial_steps"] == 0
    assert result["added_steps"] == 2
    assert result["actual_steps"] == 2
    assert result["resume_from"] is None
    assert result["seed"] == 4
    assert result["record_every"] == 500
    assert result["evaluation"] is None
    assert result["behavior_cloning"] is None
    assert result["behavior_cloning_artifact"] is None
    assert result["best_checkpoint"] is None
    assert result["best_tracking_figure"] is None
    assert result["evaluation_history"] is None
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
    assert metadata["algorithm_kwargs"] == get_algorithm("sac").effective_kwargs(
        steps=2, values=SMALL_POLICY
    )
    assert metadata["environment"]["observation_shape"] == [6]
    assert metadata["environment"]["action_shape"] == [2]
    curve = json.loads(
        (output / "training_curve.json").read_text(encoding="utf-8")
    )
    assert curve["schema_version"] == "aiogym.training_curve.v2"
    assert curve["initial_steps"] == 0
    assert [row["end_step"] for row in curve["records"]] == [2]
    assert [row["transition_count"] for row in curve["records"]] == [2]
    with zipfile.ZipFile(output / "model.zip") as checkpoint:
        assert set(checkpoint.namelist()) == {"manifest.json", "payload.zip"}
        manifest = json.loads(checkpoint.read("manifest.json"))
    assert manifest["schema_version"] == "aiogym.checkpoint.v2"
    assert manifest["algorithm"] == "sac"
    assert manifest["environment"] == metadata["environment"]
    assert manifest["runtime"]["model_class"] == "stable_baselines3.sac.sac:SAC"
    assert manifest["policy"]["training"] == {
        "completed_steps": 2,
        "seed": 4,
        "algorithm_kwargs": result["algorithm_kwargs"],
        "dataset": None,
    }
    ET.parse(output / "training_curve.svg")


def test_sac_training_continues_optimizer_and_replay_state(tmp_path):
    env = make_env("quadruple")
    config = {
        **SMALL_POLICY,
        "batch_size": 2,
        "buffer_size": 32,
        "learning_starts": 0,
    }
    first_output = tmp_path / "first"
    continued_output = tmp_path / "continued"
    try:
        first = train(
            env=env,
            algorithm="sac",
            steps=3,
            seed=6,
            algorithm_kwargs=config,
            record_every=2,
            output=first_output,
        )
        _, _, first_model, first_state = load_training_checkpoint(
            first["checkpoint"], env=env, algorithm="sac"
        )
        first_parameters = [
            value.detach().cpu().clone()
            for value in first_model.actor.parameters()
        ]

        continued = train(
            env=env,
            algorithm="sac",
            steps=2,
            resume_from=first["checkpoint"],
            record_every=1,
            output=continued_output,
        )
        _, _, continued_model, continued_state = load_training_checkpoint(
            continued["checkpoint"], env=env, algorithm="sac"
        )
    finally:
        env.close()

    curve = json.loads(
        (continued_output / "training_curve.json").read_text(encoding="utf-8")
    )
    assert first_state["completed_steps"] == 3
    assert continued["initial_steps"] == 3
    assert continued["added_steps"] == 2
    assert continued["actual_steps"] == 5
    assert continued["resume_from"] == str(
        (first_output / "model.zip").resolve()
    )
    assert continued["algorithm_kwargs"] == first["algorithm_kwargs"]
    assert continued_state["completed_steps"] == 5
    assert continued_model.num_timesteps == 5
    assert continued_model.replay_buffer.size() == 5
    assert curve["initial_steps"] == 3
    assert [row["start_step"] for row in curve["records"]] == [3, 4]
    assert [row["end_step"] for row in curve["records"]] == [4, 5]
    assert any(
        not np.array_equal(before.numpy(), after.detach().cpu().numpy())
        for before, after in zip(first_parameters, continued_model.actor.parameters())
    )


def test_continued_training_rejects_changed_seed_or_configuration(tmp_path):
    env = make_env("quadruple")
    randomized_env = make_env("quadruple", randomize=True)
    try:
        result = train(
            env=env,
            algorithm="sac",
            steps=1,
            seed=3,
            algorithm_kwargs=SMALL_POLICY,
            output=tmp_path / "first",
        )
        with pytest.raises(ValueError, match="seed must match"):
            train(
                env=env,
                algorithm="sac",
                steps=1,
                seed=4,
                resume_from=result["checkpoint"],
                output=tmp_path / "seed-mismatch",
            )
        with pytest.raises(ValueError, match="algorithm_kwargs must match"):
            train(
                env=env,
                algorithm="sac",
                steps=1,
                algorithm_kwargs={"policy_kwargs": {"net_arch": [4, 4]}},
                resume_from=result["checkpoint"],
                output=tmp_path / "config-mismatch",
            )
        with pytest.raises(ValueError, match="checkpoint algorithm"):
            train(
                env=env,
                algorithm="td3",
                steps=1,
                resume_from=result["checkpoint"],
                output=tmp_path / "algorithm-mismatch",
            )
        with pytest.raises(ValueError, match="checkpoint randomize"):
            train(
                env=randomized_env,
                algorithm="sac",
                steps=1,
                resume_from=result["checkpoint"],
                output=tmp_path / "environment-mismatch",
            )
    finally:
        randomized_env.close()
        env.close()


def test_checkpoint_rejects_incompatible_model_parameters(tmp_path):
    training_env = make_env("quadruple")
    incompatible_env = make_env(
        "quadruple",
        parameters={"pump_gain": [3.2, 3.2]},
    )
    benchmark_env = make_env("quadruple", benchmark="tracking")
    try:
        result = train(
            env=training_env,
            algorithm="sac",
            steps=2,
            algorithm_kwargs=SMALL_POLICY,
            output=tmp_path / "training",
        )
        with pytest.raises(ValueError, match="checkpoint parameters"):
            load_policy(result["checkpoint"], env=incompatible_env)
        assert isinstance(load_policy(result["checkpoint"], env=benchmark_env), aiogym.Policy)
    finally:
        benchmark_env.close()
        incompatible_env.close()
        training_env.close()


def test_training_periodically_evaluates_and_saves_best_checkpoint(tmp_path):
    env = make_env("quadruple")
    evaluation_env = make_env("quadruple")
    output = tmp_path / "evaluated-sac"
    try:
        result = train(
            env=env,
            algorithm="sac",
            steps=2,
            seed=0,
            algorithm_kwargs=SMALL_POLICY,
            evaluation_env=evaluation_env,
            evaluate_every=1,
            evaluation_seed=7,
            output=output,
        )
    finally:
        evaluation_env.close()
        env.close()

    history = json.loads(
        (output / "evaluation_history.json").read_text(encoding="utf-8")
    )
    assert history["schema_version"] == "aiogym.training_evaluation.v1"
    assert [row["step"] for row in history["records"]] == [0, 1, 2]
    assert history["ranking_metric"] == {
        "name": "return",
        "direction": "maximize",
    }
    assert history["selection_order"] == [
        {"name": "safe_completion", "direction": "maximize"},
        {"name": "episode_length", "direction": "maximize"},
        {"name": "return", "direction": "maximize"},
    ]
    assert (output / "best" / "model.zip").is_file()
    tracking_figure = output / "best" / "tracking.svg"
    ET.parse(tracking_figure)
    tracking_svg = tracking_figure.read_text(encoding="utf-8")
    assert "quadruple SAC best policy - fixed training case" in tracking_svg
    assert "Output: lower_tank_1_level [cm]" in tracking_svg
    assert "Applied action: pump_1_voltage [normalized_voltage]" in tracking_svg
    assert result["best_checkpoint"] == str(
        (output / "best" / "model.zip").resolve()
    )
    assert result["best_tracking_figure"] == str(tracking_figure.resolve())
    assert result["evaluation"]["seed"] == 7
    assert result["evaluation"]["tracking_figure"] == "best/tracking.svg"

    continuation_env = make_env("quadruple")
    try:
        continued = train(
            env=continuation_env,
            algorithm="sac",
            steps=1,
            resume_from=result["best_checkpoint"],
            output=tmp_path / "continued-best",
        )
    finally:
        continuation_env.close()
    assert continued["initial_steps"] == result["evaluation"]["best_step"]
    assert continued["actual_steps"] == continued["initial_steps"] + 1


def test_training_evaluation_never_prefers_short_unsafe_episode():
    safe = {"terminated": False, "episode_length": 600, "value": 200.0}
    unsafe = {"terminated": True, "episode_length": 300, "value": 50.0}
    assert not _is_better_training_evaluation(unsafe, safe, "minimize")
    assert _is_better_training_evaluation(safe, unsafe, "minimize")


@pytest.mark.parametrize("variation", ("disturbance", "noise"))
def test_training_evaluation_requires_deterministic_environment(tmp_path, variation):
    env = make_env("quadruple")
    evaluation_env = make_env("quadruple", **{variation: True})
    try:
        with pytest.raises(ValueError, match=f"must not enable {variation}"):
            train(
                env=env,
                algorithm="sac",
                steps=2,
                evaluation_env=evaluation_env,
                evaluate_every=1,
                output=tmp_path / "invalid-evaluation",
            )
    finally:
        evaluation_env.close()
        env.close()


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


def test_three_tank_sac_config_matches_experimental_training_configuration():
    assert THREE_TANK_SAC_CONFIG == {
        "batch_size": 256,
        "buffer_size": 100000,
        "gamma": 0.9995,
        "learning_rate": 0.0003,
        "learning_starts": 5000,
        "n_steps": 10,
        "policy_kwargs": {"net_arch": [256, 256]},
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


def test_external_sb3_algorithm_class_uses_the_complete_workflow(tmp_path):
    from stable_baselines3 import A2C

    aiogym.register_sb3_algorithm("a2c", A2C)
    env = make_env("quadruple")
    try:
        result = train(
            env=env,
            algorithm="a2c",
            steps=2,
            algorithm_kwargs={"n_steps": 2, **SMALL_POLICY},
            output=tmp_path / "a2c",
        )
        policy = load_policy(result["checkpoint"], env=env)
        observation, _ = env.reset(seed=0)
        action = policy.act(observation, {})
    finally:
        env.close()
        del algorithm_registry._BACKENDS["a2c"]
    assert result["algorithm"] == "a2c"
    assert result["actual_steps"] == 2
    assert action.shape == env.action_space.shape
    assert np.isfinite(action).all()


def test_sb3_registration_rejects_non_algorithm_class():
    with pytest.raises(TypeError, match="BaseAlgorithm subclass"):
        aiogym.register_sb3_algorithm("not_sb3", object)
    assert "not_sb3" not in aiogym.list_algorithms()


@pytest.mark.parametrize("algorithm", ("ddpg", "ppo", "sac", "td3"))
def test_supported_algorithm_constructors(algorithm):
    env = make_env("quadruple")
    try:
        backend = get_algorithm(algorithm)
        kwargs = backend.effective_kwargs(steps=2, values=SMALL_POLICY)
        model = backend.create(env=env, seed=0, algorithm_kwargs=kwargs)
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

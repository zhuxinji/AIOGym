from __future__ import annotations

import numpy as np
import pytest
from gymnasium import Wrapper

import aiogym
from aiogym import load_track
from aiogym._internal.validation import seed_sequence
from aiogym.benchmarks.evaluation import evaluate_policy_on_track
from aiogym.evaluation.metrics.robustness import paired_seed_metadata
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.validation import ValidationEpisodePlan


TRACK_ID = "quadruple-regulation-generalist-v1"


@pytest.mark.parametrize("values", ([True], [-1], [1, 1], []))
def test_public_seed_entrypoints_reject_invalid_values(values):
    track = load_track(TRACK_ID)
    with pytest.raises((TypeError, ValueError)):
        seed_sequence("seeds", values)
    with pytest.raises((TypeError, ValueError)):
        ValidationEpisodePlan(track, base_seeds=values)
    with pytest.raises((TypeError, ValueError)):
        paired_seed_metadata(values)
    with pytest.raises((TypeError, ValueError)):
        evaluate_policy_on_track(object(), track, base_seeds=values)
    with pytest.raises((TypeError, ValueError)):
        RLTrainingConfig(
            track_id=TRACK_ID,
            algorithm_id="sac",
            training_seed=0,
            total_transitions=1,
            n_envs=1,
            validation_seeds=values,
        )


def test_numpy_integer_seeds_are_normalized():
    seeds = seed_sequence("seeds", [np.int64(1), np.int32(2)])
    assert seeds == (1, 2)
    assert all(type(seed) is int for seed in seeds)
    config = RLTrainingConfig(
        track_id=TRACK_ID,
        algorithm_id="sac",
        training_seed=0,
        total_transitions=1,
        n_envs=1,
        validation_seeds=(np.int64(3),),
    )
    assert config.validation_seeds == (3,)
    assert type(config.validation_seeds[0]) is int


def _evaluation_env(*, info_level="full"):
    return aiogym.make_env(
        config={
            "scenario": "cstr",
            "reward_spec": "regulation-v1",
            "info_level": info_level,
            "environment": {"episode_steps": 1, "auto_events": False},
        }
    )


@pytest.mark.parametrize("values", ([], [True], [-1], [1, 1]))
def test_public_evaluate_controller_rejects_invalid_seed_list(values):
    env = _evaluation_env()
    controller = aiogym.make_controller("pid", scenario="cstr")
    try:
        with pytest.raises((TypeError, ValueError)):
            aiogym.evaluate_controller(controller, env, seed_list=values)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"episodes": 0}, "episodes"),
        ({"episodes": True}, "episodes"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
    ),
)
def test_public_evaluate_controller_rejects_invalid_scalar_seed_inputs(
    kwargs,
    message,
):
    env = _evaluation_env()
    controller = aiogym.make_controller("pid", scenario="cstr")
    try:
        with pytest.raises(ValueError, match=message):
            aiogym.evaluate_controller(controller, env, **kwargs)
    finally:
        env.close()


def test_public_evaluate_controller_normalizes_numpy_seed_list():
    env = _evaluation_env()
    controller = aiogym.make_controller("pid", scenario="cstr")
    try:
        result = aiogym.evaluate_controller(
            controller,
            env,
            episodes=True,
            seed=True,
            seed_list=[np.int64(1), np.int32(2)],
        )
    finally:
        env.close()
    assert result["episodes"] == 2
    assert result["seed"] == 1
    assert result["seed_list"] == [1, 2]
    assert all(type(seed) is int for seed in result["seed_list"])


class _ResetTrapWrapper(Wrapper):
    def reset(self, *args, **kwargs):
        raise AssertionError("minimal-info guard must run before reset")


@pytest.mark.parametrize("wrapped", (False, True))
def test_public_evaluate_controller_rejects_minimal_info_before_reset(wrapped):
    env = _evaluation_env(info_level="minimal")
    if wrapped:
        env = _ResetTrapWrapper(env)
    else:
        env.reset = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("minimal-info guard must run before reset")
        )
    controller = aiogym.make_controller("pid", scenario="cstr")
    try:
        with pytest.raises(ValueError, match="info_level='full'"):
            aiogym.evaluate_controller(controller, env)
    finally:
        env.close()


def test_public_evaluate_controller_rejects_non_aiogym_environment():
    with pytest.raises(TypeError, match="AIO-Gym environment"):
        aiogym.evaluate_controller(object(), object())

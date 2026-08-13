from __future__ import annotations

import numpy as np

from aiogym import DatasetReader, collect, evaluate, make_env
from aiogym.scenarios.three_tank.wrappers import Tank3ResidualWrapper


def _randomized_env():
    return make_env(
        "three_tank",
        reward="tank3-regulation",
        randomize=True,
        noise=True,
        delay=True,
        fault={"probability": 1.0},
    )


def test_randomization_is_seeded_and_records_resolved_episode_and_channels():
    first = _randomized_env()
    second = _randomized_env()
    different = _randomized_env()
    try:
        first_observation, first_info = first.reset(seed=41)
        second_observation, second_info = second.reset(seed=41)
        different_observation, different_info = different.reset(seed=42)
        assert np.array_equal(first_observation, second_observation)
        assert first_info["episode_spec"] == second_info["episode_spec"]
        assert first_info["runtime_variation"] == second_info["runtime_variation"]
        assert not np.array_equal(first_observation, different_observation)
        assert first_info["episode_spec"] != different_info["episode_spec"]
        assert first_info["episode_family"] in {
            "tracking",
            "disturbance-rejection",
            "boundary-safety",
        }
        variation = first_info["runtime_variation"]
        assert 0 <= variation["action_delay_steps"] <= 1
        assert 0 <= variation["observation_delay_steps"] <= 2
        assert variation["fault"]["kind"] == "loss-of-effectiveness"
    finally:
        first.close()
        second.close()
        different.close()


def test_episode_sampling_rng_is_independent_from_channel_options():
    plain = make_env("three_tank", randomize=True)
    varied = make_env(
        "three_tank", randomize=True, noise=True, delay=True, fault=True
    )
    try:
        _, plain_info = plain.reset(seed=19)
        _, varied_info = varied.reset(seed=19)
    finally:
        plain.close()
        varied.close()
    assert plain_info["episode_family"] == varied_info["episode_family"]
    assert plain_info["episode_spec"] == varied_info["episode_spec"]


def test_base_environment_does_not_randomize_automatically():
    env = make_env("three_tank", reward="tank3-regulation")
    try:
        first_observation, first_info = env.reset(seed=1)
        second_observation, second_info = env.reset(seed=2)
    finally:
        env.close()
    assert np.array_equal(first_observation, second_observation)
    assert first_info["episode_spec"] == second_info["episode_spec"]
    assert first_info["episode_family"] == second_info["episode_family"] == "default"


def test_sampled_tracking_reference_event_is_episode_specific():
    env = make_env("three_tank", randomize=True)
    try:
        for seed in range(100):
            _, reset_info = env.reset(seed=seed)
            if reset_info["episode_family"] == "tracking":
                break
        else:
            raise AssertionError("no tracking episode sampled")
        schedule = reset_info["episode_spec"]["reference_schedule"]
        step_text, target = next(iter(schedule.items()))
        env.unwrapped._step_index = int(step_text)
        env.unwrapped._apply_events()
        assert np.allclose(env.unwrapped.y_sp, target)
    finally:
        env.close()


def test_randomized_residual_dataset_keeps_episode_metadata(tmp_path):
    env = Tank3ResidualWrapper(
        make_env("three_tank", reward="tank3-regulation", randomize=True)
    )
    try:
        result = collect(
            env=env,
            policy="random",
            episodes=1,
            seed=17,
            max_steps=2,
            output=tmp_path / "tank3-dataset",
        )
    finally:
        env.close()
    episode = DatasetReader(result["path"])[0]
    assert episode.metadata["episode_family"] in {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert episode.metadata["episode_spec"]["horizon"] > 0
    assert episode.array("action").shape == (2, 2)
    assert episode.array("channel_action").shape == (2, 5)


def test_randomized_residual_environment_uses_the_standard_evaluator():
    env = Tank3ResidualWrapper(
        make_env("three_tank", reward="tank3-regulation", randomize=True)
    )
    try:
        result = evaluate(
            env=env,
            policy="random",
            seeds=[3, 4],
            max_steps=2,
        )
    finally:
        env.close()
    assert result["seeds"] == [3, 4]
    assert [episode["length"] for episode in result["episodes"]] == [2, 2]
    assert "tank3_tracking_iae" in result["aggregate"]

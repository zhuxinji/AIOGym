from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import aiogym


@pytest.mark.parametrize(
    "noise",
    (
        {"std": 0.02, "bias_std": 0.0},
        {"std": 0.0, "bias_std": 0.02},
        {"std": 5.0, "bias_std": 0.0},
    ),
)
def test_cstr_noisy_errors_come_from_the_returned_measurements(noise):
    clean = aiogym.make_env("cstr")
    noisy = aiogym.make_env("cstr", noise=noise)
    try:
        observation, info = noisy.reset(seed=17)
        clean_observation, clean_info = clean.reset(seed=17)
        action = np.asarray(clean.unwrapped.model.default_action(), dtype=np.float32)
        for _ in range(2):
            assert noisy.observation_space.contains(observation)
            measurement = observation[:2].astype(float) * [1.5, 200.0]
            reference = [0.02, 45.0] + observation[2:4].astype(float) * [0.18, 45.0]
            expected = np.clip((measurement - reference) / [0.18, 45.0], -1.0, 1.0)
            np.testing.assert_allclose(observation[4:], expected, rtol=0, atol=6e-8)
            np.testing.assert_array_equal(observation[2:4], clean_observation[2:4])
            np.testing.assert_array_equal(info["y"], clean_info["y"])
            # Errors inherit sensor bias; they have no separately injected bias.
            np.testing.assert_array_equal(
                info["runtime_variation"]["observation_bias"][4:], [0.0, 0.0]
            )
            assert not np.array_equal(observation[:2], clean_observation[:2])
            assert not np.array_equal(observation[4:], clean_observation[4:])
            clean_observation, clean_reward, clean_done, clean_cut, clean_info = (
                clean.step(action)
            )
            observation, reward, done, cut, info = noisy.step(action)
            assert (reward, done, cut) == (clean_reward, clean_done, clean_cut)
    finally:
        clean.close()
        noisy.close()


def test_cstr_zero_noise_preserves_clean_observations_exactly():
    clean = aiogym.make_env("cstr")
    candidate = aiogym.make_env("cstr", noise={"std": 0.0, "bias_std": 0.0})
    try:
        expected, _ = clean.reset(seed=17)
        actual, _ = candidate.reset(seed=17)
        action = np.asarray(clean.unwrapped.model.default_action(), dtype=np.float32)
        for _ in range(2):
            np.testing.assert_array_equal(actual, expected)
            expected, *_ = clean.step(action)
            actual, *_ = candidate.step(action)
    finally:
        clean.close()
        candidate.close()


def test_cstr_noisy_errors_use_the_same_delayed_reference_as_the_observation():
    env = aiogym.make_env(
        "cstr", noise=True, delay={"observation_steps": 2, "action_steps": 0}
    )
    try:
        episode = replace(
            env.unwrapped.default_episode,
            reference_schedule={1: (0.10, 60.0)},
        )
        observation, _ = env.reset(seed=17, options={"episode": episode})
        old_reference = observation[2:4].copy()
        action = np.asarray(env.unwrapped.model.default_action(), dtype=np.float32)
        observations = []
        for _ in range(3):
            observation, _, _, _, info = env.step(action)
            observations.append(observation)
            measurement = observation[:2].astype(float) * [1.5, 200.0]
            reference = [0.02, 45.0] + observation[2:4].astype(float) * [0.18, 45.0]
            np.testing.assert_allclose(info["policy_reference"], reference)
            np.testing.assert_array_equal(info["reference"], [0.10, 60.0])
            expected = np.clip((measurement - reference) / [0.18, 45.0], -1.0, 1.0)
            np.testing.assert_allclose(observation[4:], expected, rtol=0, atol=6e-8)
        np.testing.assert_array_equal(observations[0][2:4], old_reference)
        np.testing.assert_array_equal(observations[1][2:4], old_reference)
        np.testing.assert_allclose(
            observations[2][2:4], [(0.10 - 0.02) / 0.18, (60.0 - 45.0) / 45.0]
        )
    finally:
        env.close()

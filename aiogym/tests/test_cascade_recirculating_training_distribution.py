from __future__ import annotations

import json
import math

import numpy as np

from aiogym.benchmarks import load_track
from aiogym.generation import (
    CascadeRecirculatingTrainingSampler,
    load_distribution,
    validate_cascade_recirculating_episode,
)
from aiogym.models import apply_model_params, make_model
from aiogym.rl.episode_env import make_track_training_base_env


TRACK_ID = "cascade-recirculating-regulation-generalist-v1"


def test_recirculating_l0_l2_are_registered_and_deterministic():
    for level in ("l0", "l1", "l2"):
        distribution_id = (
            "cascade-recirculating-regulation-training-"
            f"{level}-v1"
        )
        distribution = load_distribution(distribution_id)
        assert distribution.scenario_id == "cascade-recirculating"
        first = CascadeRecirculatingTrainingSampler(distribution).sample(91)
        second = CascadeRecirculatingTrainingSampler(distribution).sample(91)
        assert first.as_dict() == second.as_dict()


def test_recirculating_distribution_is_feasible_for_100_fixed_seeds():
    sampler = CascadeRecirculatingTrainingSampler(level="L2")
    for seed in range(100):
        episode = sampler.sample(seed)
        report = validate_cascade_recirculating_episode(episode)
        assert report["passed"], (seed, report)
        model = apply_model_params(
            make_model("cascade-recirculating"),
            episode.plant_parameters,
        )
        assert len(episode.initial_state) == len(model.initial_state())
        assert all(math.isfinite(value) for value in episode.initial_state)
        for value, row in zip(
            episode.initial_state,
            model.state_schema(),
        ):
            lower, upper = row["bounds"]
            assert lower <= value <= upper
        assert episode.reference_schedule[0]["at_step"] == 0
        assert all(
            0 <= event["at_step"] < episode.episode_steps
            for event in (
                *episode.reference_schedule,
                *episode.disturbance_schedule,
            )
        )


def test_recirculating_l2_covers_multiple_realizations():
    sampler = CascadeRecirculatingTrainingSampler(level="L2")
    episodes = [sampler.sample(seed) for seed in range(24)]

    def identities(selector):
        return {
            json.dumps(selector(episode), sort_keys=True)
            for episode in episodes
        }

    assert len(identities(lambda row: row.plant_parameters)) > 1
    assert len(identities(lambda row: row.initial_state)) > 1
    assert len(identities(lambda row: row.reference_schedule)) > 1
    assert len(identities(lambda row: row.disturbance_schedule)) > 1


def test_recirculating_track_contract_and_short_rollout_are_finite():
    track = load_track(TRACK_ID)
    distribution = track.training_distribution()
    sampler = CascadeRecirculatingTrainingSampler(distribution)
    env = make_track_training_base_env(track, sampler=sampler)
    try:
        assert distribution.control_dt == track.policy_contract["control_dt"]
        assert distribution.goal == track.goal
        assert env.unwrapped.episode_steps == distribution.episode_steps
        for seed in range(3):
            episode = sampler.sample(seed)
            observation, _ = env.reset(
                seed=episode.base_seed,
                options={"episode_spec": episode},
            )
            assert observation.shape == env.observation_space.shape
            assert np.all(np.isfinite(observation))
            for _ in range(20):
                action = np.asarray(
                    env.unwrapped.model.default_action(),
                    dtype=np.float32,
                )
                assert action.shape == env.action_space.shape
                observation, reward, terminated, truncated, _ = env.step(
                    action
                )
                assert np.all(np.isfinite(observation))
                assert math.isfinite(float(reward))
                assert not terminated
                assert not truncated
    finally:
        env.close()

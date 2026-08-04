from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from aiogym.benchmarks import load_track
from aiogym.benchmarks.tracks.schema import TrackSpec
from aiogym.generation import load_distribution
from aiogym.generation.cascade import (
    CASCADE_PRODUCT_FLOW_M3S,
    CascadeTrainingSampler,
    cascade_case_conditioned_training_distribution,
    validate_cascade_episode,
)
from aiogym.models.registry import apply_model_params, make_model
from aiogym.rl.episode_env import make_track_training_base_env


TRACK_ID = "cascade-regulation-generalist-v1"
V2_TRACK_ID = "cascade-regulation-generalist-v2"


def test_cascade_l0_l2_are_registered_and_deterministic():
    for level in ("l0", "l1", "l2"):
        distribution_id = f"cascade-regulation-training-{level}-v1"
        distribution = load_distribution(distribution_id)
        assert distribution.scenario_id == "cascade"
        first = CascadeTrainingSampler(distribution).sample(91)
        second = CascadeTrainingSampler(distribution).sample(91)
        assert first.as_dict() == second.as_dict()


def test_cascade_v1_distribution_identity_remains_immutable():
    distribution = load_distribution("cascade-regulation-training-l2-v1")
    assert distribution.distribution_hash == (
        "1032928566d244c5bddac8e94254b8edb711e2a8993709814db3f2d25a8d2c11"
    )


def test_case_conditioned_v2_covers_both_declared_training_modes():
    distribution = cascade_case_conditioned_training_distribution()
    registered = load_distribution("cascade-regulation-training-l2-v2")
    assert registered == distribution
    assert distribution.episode_steps == 2640
    assert distribution.declaration["mixture_weights"] == {
        "commissioning": 0.5,
        "temperature-step": 0.5,
    }

    sampler = CascadeTrainingSampler(distribution)
    episodes = [sampler.sample(91, episode_index=index) for index in range(200)]
    by_mode = {
        mode: [episode for episode in episodes if mode in episode.difficulty_tags]
        for mode in ("commissioning", "temperature-step")
    }
    assert all(by_mode.values())
    assert {
        len(episode.reference_schedule)
        for episode in by_mode["commissioning"]
    } == {3}
    assert {
        len(episode.reference_schedule)
        for episode in by_mode["temperature-step"]
    } == {5}
    assert all(
        validate_cascade_episode(episode)["passed"] for episode in episodes
    )
    assert len({episode.resolved_hash for episode in episodes}) == len(episodes)


def test_cascade_v2_track_binds_case_conditioned_distribution_weights():
    track = load_track(V2_TRACK_ID)
    distribution = track.training_distribution()
    declared_weights = {
        case.case_id.split(":", 1)[0]: case.weight
        for case in track.resolved_cases("training")
    }
    total = sum(declared_weights.values())
    normalized = {
        case_id: weight / total
        for case_id, weight in declared_weights.items()
    }
    assert track.train_distribution_id == (
        "cascade-regulation-training-l2-v2"
    )
    assert distribution.declaration["mixture_weights"] == normalized


def test_cascade_v2_rejects_track_distribution_weight_drift():
    declaration = deepcopy(
        load_track(V2_TRACK_ID, validate_policy_contract=False).declaration
    )
    declaration["training"]["cases"][0]["weight"] = 3.0
    track = TrackSpec(declaration)
    with pytest.raises(
        ValueError,
        match="distribution weights do not match",
    ):
        track.training_distribution()


def test_cascade_distribution_is_feasible_for_100_fixed_seeds():
    sampler = CascadeTrainingSampler(level="L2")
    for seed in range(100):
        episode = sampler.sample(seed)
        report = validate_cascade_episode(episode)
        assert report["passed"], (seed, report)
        model = apply_model_params(
            make_model("cascade"),
            episode.plant_parameters,
        )
        model.configure_operation(
            {
                "product_flow_sp": CASCADE_PRODUCT_FLOW_M3S,
                "min_product_flow": CASCADE_PRODUCT_FLOW_M3S,
            }
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


def test_cascade_l2_covers_multiple_realizations():
    sampler = CascadeTrainingSampler(level="L2")
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


def test_cascade_track_contract_and_short_rollout_are_finite():
    track = load_track(TRACK_ID)
    distribution = track.training_distribution()
    sampler = CascadeTrainingSampler(distribution)
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

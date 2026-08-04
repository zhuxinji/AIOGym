"""Static and one-step contracts for split-isolated regulation v2 Tracks."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from aiogym._environment.builder import build_track_case_environment
from aiogym.benchmarks.tracks import (
    TrackSpec,
    audit_split_isolation,
    load_track,
    require_split_isolation,
)
from aiogym.rl.episode_env import make_track_episode_sampler


V2_TRACK_IDS = (
    "quadruple-regulation-generalist-v2",
    "cascade-regulation-generalist-v2",
    "cascade-recirculating-regulation-generalist-v2",
)
V1_HASHES = {
    "quadruple-regulation-generalist-v1": (
        "112931a0adab0e01d6d4c26c4187ad05a84d8cf1e6ac1c886350915068d284c6"
    ),
    "cascade-regulation-generalist-v1": (
        "e078072c8b61e68de84393af360613923fa295b729fa4b94d07a62ed95a957ca"
    ),
}


@pytest.mark.parametrize("track_id", V2_TRACK_IDS)
def test_v2_track_loads_and_has_frozen_protocol_identity(track_id):
    track = load_track(track_id)
    scenario = track.scenario
    declaration = track.declaration
    assert track.official
    assert declaration["ranking"]["id"] == (
        f"{scenario}-regulation-ranking-v2"
    )
    assert declaration["ranking"]["anchor_id"] == (
        f"{scenario}-regulation-anchors-v3"
    )
    assert "step_budget" not in declaration["training"]
    for split in ("training", "validation", "test"):
        assert track.seed_namespace(split) == (
            f"{scenario}-regulation-{split}-v2"
        )


@pytest.mark.parametrize("track_id", V2_TRACK_IDS)
def test_v2_validation_and_test_profiles_are_split_isolated(track_id):
    report = audit_split_isolation(track_id)
    assert report["passed"] is True
    assert report["overlapping_hashes"] == []
    require_split_isolation(track_id)


@pytest.mark.parametrize("track_id,expected_hash", V1_HASHES.items())
def test_v1_track_hash_and_legacy_overlap_remain_compatible(
    track_id,
    expected_hash,
):
    track = load_track(track_id)
    report = audit_split_isolation(track)
    assert track.track_hash == expected_hash
    assert report["passed"] is False
    assert report["overlapping_hashes"]
    with pytest.raises(ValueError, match="overlap"):
        require_split_isolation(track)


def test_only_changing_seed_namespace_does_not_create_split_isolation():
    source = load_track(
        "quadruple-regulation-generalist-v1"
    ).declaration
    declaration = deepcopy(source)
    declaration["id"] = "namespace-only-isolation-is-invalid-v2"
    declaration["validation"]["seed_spec"]["namespace"] = (
        "unique-validation-v2"
    )
    declaration["test"]["seed_spec"]["namespace"] = "unique-test-v2"
    report = audit_split_isolation(TrackSpec(declaration))
    assert report["passed"] is False
    assert len(report["overlapping_hashes"]) == 2


@pytest.mark.parametrize("track_id", V2_TRACK_IDS)
def test_v2_variant_ids_and_resolved_hashes_are_unique(track_id):
    track = load_track(track_id)
    cases = [
        case
        for split in ("validation", "test")
        for case in track.resolved_cases(split)
    ]
    variant_keys = [
        (case.profile["name"], case.variant_id) for case in cases
    ]
    hashes = [case.resolved_case_hash for case in cases]
    assert all(case.variant_id for case in cases)
    assert len(variant_keys) == len(set(variant_keys))
    assert len(hashes) == len(set(hashes))


@pytest.mark.parametrize("track_id", V2_TRACK_IDS)
def test_v2_case_events_are_inside_each_episode_horizon(track_id):
    track = load_track(track_id)
    for split in ("validation", "test"):
        for case in track.resolved_cases(split):
            profile = case.profile
            horizon = int(profile["environment"]["episode_steps"])
            events = [
                *profile.get("setpoints", {}).get("schedule", ()),
                *profile.get("disturbances", ()),
            ]
            assert all(0 <= int(event["at_step"]) < horizon for event in events)


def test_cascade_v2_observation_contract_is_normalized_and_unmeasured():
    track = load_track("cascade-regulation-generalist-v2")
    actual = track.validate_policy_contract()
    assert actual["normalize_observations"] is True
    assert actual["disturbance_obs"] is False
    assert not any(
        feature.startswith("disturbance:")
        for feature in actual["observation_features"]
    )
    case = track.resolved_cases("validation")[0]
    env = build_track_case_environment(track, case)
    try:
        assert np.all(np.isfinite(env.observation_space.low))
        assert np.all(np.isfinite(env.observation_space.high))
        observation, _ = env.reset(seed=0)
        neutral = 0.5 * (env.action_space.low + env.action_space.high)
        next_observation, reward, terminated, truncated, _ = env.step(neutral)
        assert env.observation_space.contains(observation)
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_cascade_v2_fixed_regulation_targets_are_steady_state_feasible():
    track = load_track("cascade-regulation-generalist-v2")
    for split in ("validation", "test"):
        for case in track.resolved_cases(split):
            if case.profile["name"] == "disturbance-rejection":
                continue
            env = build_track_case_environment(track, case)
            try:
                setpoints = case.profile["setpoints"]
                targets = [
                    setpoints["initial"],
                    *(
                        event["values"]
                        for event in setpoints.get("schedule", ())
                    ),
                ]
                assert all(
                    env.model.steady_state_requirements(target)["feasible"]
                    for target in targets
                )
            finally:
                env.close()


def test_quadruple_v2_keeps_v1_policy_facing_physical_semantics():
    v1 = load_track("quadruple-regulation-generalist-v1")
    v2 = load_track("quadruple-regulation-generalist-v2")
    assert v2.policy_contract == v1.policy_contract
    first = v1.validate_policy_contract()
    second = v2.validate_policy_contract()
    for key in (
        "action_shape",
        "action_features",
        "observation_shape",
        "observation_features",
        "controlled_output_shape",
        "controlled_output_features",
    ):
        assert second[key] == first[key]
    v1_env = build_track_case_environment(
        v1,
        v1.resolved_cases("validation")[0],
    )
    v2_env = build_track_case_environment(
        v2,
        v2.resolved_cases("validation")[0],
    )
    try:
        assert not np.all(np.isfinite(v1_env.observation_space.low))
        assert np.all(np.isfinite(v2_env.observation_space.low))
        assert np.all(np.isfinite(v2_env.observation_space.high))
    finally:
        v1_env.close()
        v2_env.close()


@pytest.mark.parametrize("track_id", V2_TRACK_IDS)
def test_training_sampler_identity_is_separate_from_fixed_holdouts(track_id):
    track = load_track(track_id)
    sampler = make_track_episode_sampler(track)
    episodes = [sampler.sample(7, episode_index=index) for index in range(3)]
    fixed_hashes = {
        case.resolved_case_hash
        for split in ("validation", "test")
        for case in track.resolved_cases(split)
    }
    assert sampler.distribution_id == track.train_distribution_id
    assert len({episode.resolved_hash for episode in episodes}) == 3
    assert all(episode.resolved_hash not in fixed_hashes for episode in episodes)


@pytest.mark.parametrize("track_id", V2_TRACK_IDS)
def test_v2_hidden_disturbance_pair_shares_non_disturbance_context(track_id):
    track = load_track(track_id)
    pair = [
        case
        for case in track.resolved_cases("test")
        if case.pair_id is not None
    ]
    assert {case.condition for case in pair} == {"nominal", "shifted"}
    nominal = next(case for case in pair if case.condition == "nominal")
    shifted = next(case for case in pair if case.condition == "shifted")
    nominal_profile = deepcopy(dict(nominal.profile))
    shifted_profile = deepcopy(dict(shifted.profile))
    assert nominal_profile.pop("disturbances") == []
    assert shifted_profile.pop("disturbances")
    assert nominal_profile == shifted_profile
    assert "disturbance_hidden_from_controller" in (
        nominal.profile["acceptance"]["required_checks"]
    )

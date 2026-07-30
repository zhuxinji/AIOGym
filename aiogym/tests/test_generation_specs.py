from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from aiogym.generation import (
    DISTRIBUTION_SCHEMA_VERSION,
    EPISODE_SPEC_SCHEMA_VERSION,
    SEED_COMPONENTS,
    DistributionSpec,
    EpisodeSpec,
    FixedCaseEpisodeSampler,
    SeedTree,
    distribution_spec_from_case,
    episode_spec_from_case,
    seed_namespace,
)
from aiogym.tests._env import make_test_env as make_env


def test_distribution_same_declaration_has_stable_hash():
    declaration = _distribution_declaration()
    first = DistributionSpec(declaration)
    reordered = {
        key: deepcopy(declaration[key])
        for key in reversed(tuple(declaration))
    }
    second = DistributionSpec(reordered)

    assert first.distribution_hash == second.distribution_hash
    declaration["plant_distribution"]["coefficient"] = [99.0]
    assert first.as_dict()["plant_distribution"]["coefficient"] == [0.9, 1.1]


def test_distribution_rejects_non_finite_or_invalid_weights():
    declaration = _distribution_declaration()
    declaration["mixture_weights"] = {"nominal": 0.0}
    with pytest.raises(ValueError, match="positive total"):
        DistributionSpec(declaration)

    declaration = _distribution_declaration()
    declaration["plant_distribution"]["bad"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        DistributionSpec(declaration)


def test_seed_tree_same_identity_has_same_independent_component_seeds():
    first = SeedTree.for_split(
        41,
        "quadruple-regulation-generalist-v1",
        "training",
        worker_index=2,
        episode_index=7,
    )
    second = SeedTree.for_split(
        41,
        "quadruple-regulation-generalist-v1",
        "training",
        worker_index=2,
        episode_index=7,
    )

    assert first.metadata() == second.metadata()
    assert set(first.component_seeds) == set(SEED_COMPONENTS)
    assert len(set(first.component_seeds.values())) == len(SEED_COMPONENTS)
    assert first.generator("reference").integers(0, 2**31) == (
        second.generator("reference").integers(0, 2**31)
    )


def test_train_validation_test_seed_namespaces_do_not_overlap():
    trees = {
        split: SeedTree.for_split(5, "track", split)
        for split in ("training", "validation", "test")
    }
    assert len({tree.namespace for tree in trees.values()}) == 3
    all_seeds = [
        seed
        for tree in trees.values()
        for seed in tree.component_seeds.values()
    ]
    assert len(set(all_seeds)) == len(all_seeds)
    assert seed_namespace("track", "training").endswith(
        ":training:episodes:v1"
    )


def test_episode_spec_is_deterministic_and_defensively_copied():
    distribution = DistributionSpec(_distribution_declaration())
    tree = SeedTree.for_split(17, distribution.distribution_id, "training")
    parameters = {"pump_gain": 1.05}
    first = EpisodeSpec.from_distribution(
        distribution,
        base_seed=tree.base_seed,
        component_seeds=tree.component_seeds,
        plant_parameters=parameters,
        initial_state=(1.0, 2.0),
        reference_schedule=(
            {"at_step": 2, "values": [1.1, 2.1]},
        ),
        disturbance_schedule=(
            {"at_step": 5, "name": "feed", "value": 0.2},
        ),
        sensor_model={"kind": "identity"},
        actuator_model={"kind": "identity"},
        difficulty_tags=("nominal",),
    )
    second = EpisodeSpec(first.as_dict())

    assert first.as_dict() == second.as_dict()
    assert first.episode_spec_id == second.episode_spec_id
    assert first.resolved_hash == second.resolved_hash
    assert first.distribution_hash == distribution.distribution_hash
    parameters["pump_gain"] = 99.0
    assert first.as_dict()["plant_parameters"]["pump_gain"] == 1.05

    external = first.as_dict()
    external["initial_state"][0] = 99.0
    assert first.initial_state == (1.0, 2.0)


def test_episode_spec_identity_detects_payload_tampering():
    distribution = DistributionSpec(_distribution_declaration())
    tree = SeedTree.for_split(3, distribution.distribution_id, "training")
    episode = EpisodeSpec.from_distribution(
        distribution,
        base_seed=tree.base_seed,
        component_seeds=tree.component_seeds,
        plant_parameters={},
        initial_state=(1.0,),
    )
    tampered = episode.as_dict()
    tampered["initial_state"] = [2.0]
    with pytest.raises(ValueError, match="resolved_hash"):
        EpisodeSpec(tampered)


def test_episode_spec_rejects_runtime_ambiguity():
    distribution = DistributionSpec(_distribution_declaration())
    tree = SeedTree.for_split(3, distribution.distribution_id, "training")
    with pytest.raises(ValueError, match="ordered"):
        EpisodeSpec.from_distribution(
            distribution,
            base_seed=tree.base_seed,
            component_seeds=tree.component_seeds,
            plant_parameters={},
            initial_state=(1.0,),
            reference_schedule=(
                {"at_step": 4, "values": [1.0]},
                {"at_step": 2, "values": [2.0]},
            ),
        )


def test_fixed_case_resolves_deterministically():
    sampler = FixedCaseEpisodeSampler(
        "quadruple/minimum-phase",
        split="validation",
    )
    first = sampler.sample(23)
    second = sampler.sample(23)
    direct = episode_spec_from_case(
        "quadruple/minimum-phase",
        seed=23,
        split="validation",
    )

    assert first.as_dict() == second.as_dict() == direct.as_dict()
    assert first.difficulty_tags == (
        "fixed-case",
        "quadruple/minimum-phase",
    )
    assert first.reference_schedule[0]["at_step"] == 0
    assert first.distribution_hash == (
        distribution_spec_from_case(
            "quadruple/minimum-phase"
        ).distribution_hash
    )


def test_env_reset_episode_spec_bypasses_config_migration_randomization():
    episode = episode_spec_from_case(
        "quadruple/minimum-phase",
        seed=29,
        split="validation",
    )
    deterministic = make_env(
        "quadruple",
        case="minimum-phase",
    )
    randomized = make_env(
        "quadruple",
        case="minimum-phase",
        randomize=True,
        randomize_setpoints=True,
        randomize_plant=True,
        auto_events=True,
        noise=True,
    )
    assert not hasattr(randomized, "legacy_distribution_spec")
    try:
        first_obs, first_info = deterministic.reset(
            options={"episode_spec": episode}
        )
        second_obs, second_info = randomized.reset(
            options={"episode_spec": episode.as_dict()}
        )
        assert first_obs.tolist() == second_obs.tolist()
        assert first_info["episode_spec_id"] == episode.episode_spec_id
        assert second_info["episode_spec_hash"] == episode.resolved_hash
        assert second_info["distribution_hash"] == (
            episode.distribution_hash
        )

        action = deterministic.action_space.sample()
        first_transition = deterministic.step(action)
        second_transition = randomized.step(action)
        assert np.array_equal(
            first_transition[0],
            second_transition[0],
        )
        assert first_transition[1] == pytest.approx(second_transition[1])
        assert first_transition[2:4] == second_transition[2:4]
        assert (
            second_transition[4]["episode_spec_id"]
            == episode.episode_spec_id
        )
    finally:
        deterministic.close()
        randomized.close()


def test_env_rejects_episode_spec_for_different_runtime_contract():
    episode = episode_spec_from_case(
        "quadruple/minimum-phase",
        seed=31,
    )
    env = make_env(
        "quadruple",
        case="minimum-phase",
        episode_steps=10,
    )
    try:
        with pytest.raises(ValueError, match="episode_steps"):
            env.reset(options={"episode_spec": episode})
    finally:
        env.close()


def test_episode_spec_resolves_sensor_randomness_by_seed():
    sampler = FixedCaseEpisodeSampler("quadruple/minimum-phase")
    fixed = sampler.sample(37)
    declaration = sampler.distribution.declaration
    declaration["sensor_distribution"] = {
        "kind": "additive_gaussian",
        "noise_pct": 0.01,
    }
    noisy_distribution = DistributionSpec(declaration)
    noisy = EpisodeSpec.from_distribution(
        noisy_distribution,
        base_seed=fixed.base_seed,
        component_seeds=fixed.component_seeds,
        plant_parameters=fixed.plant_parameters,
        initial_state=fixed.initial_state,
        reference_schedule=fixed.reference_schedule,
        disturbance_schedule=fixed.disturbance_schedule,
        sensor_model=declaration["sensor_distribution"],
        actuator_model=fixed.actuator_model,
        economic_context=fixed.economic_context,
        difficulty_tags=fixed.difficulty_tags,
    )
    env = make_env("quadruple", case="minimum-phase")
    action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
    try:
        first_observation, _ = env.reset(
            options={"episode_spec": noisy}
        )
        first_transition = env.step(action)
        second_observation, _ = env.reset(
            options={"episode_spec": noisy}
        )
        second_transition = env.step(action)

        assert np.array_equal(first_observation, second_observation)
        assert np.array_equal(
            first_transition[0],
            second_transition[0],
        )
        assert first_transition[1:4] == second_transition[1:4]
    finally:
        env.close()


def _distribution_declaration():
    return {
        "schema_version": DISTRIBUTION_SCHEMA_VERSION,
        "distribution_id": "quadruple-regulation-train-v1",
        "scenario_id": "quadruple",
        "goal": "regulation",
        "control_dt": 1.0,
        "episode_steps": 20,
        "plant_distribution": {
            "kind": "bounded",
            "coefficient": [0.9, 1.1],
        },
        "initial_state_distribution": {
            "kind": "equilibrium_perturbation",
        },
        "reference_distribution": {"kind": "piecewise_constant"},
        "disturbance_distribution": {"kind": "step"},
        "sensor_distribution": {"kind": "identity"},
        "actuator_distribution": {"kind": "identity"},
        "economic_context_distribution": None,
        "mixture_weights": {"nominal": 0.8, "mild-shift": 0.2},
        "curriculum_id": None,
    }

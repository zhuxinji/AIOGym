from __future__ import annotations

import numpy as np
import pytest

from aiogym.tests._env import make_test_env as make_env
from aiogym.generation import (
    CURRICULUM_LEVELS,
    QUADRUPLE_CURRICULUM_V1,
    QuadrupleCurriculumSampler,
    QuadrupleDisturbanceGenerator,
    QuadrupleReferenceGenerator,
    QuadrupleTrainingSampler,
    load_distribution,
    quadruple_training_distribution,
    validate_quadruple_episode,
    validate_quadruple_parameters,
)
from aiogym.models import apply_model_params, make_model
from aiogym.benchmarks import load_track


def test_curriculum_declares_versioned_l0_through_l4():
    assert tuple(CURRICULUM_LEVELS) == ("L0", "L1", "L2", "L3", "L4")
    assert (
        QUADRUPLE_CURRICULUM_V1.level_for_transition(0).level_id
        == "L0"
    )
    assert (
        QUADRUPLE_CURRICULUM_V1.level_for_transition(599_999).level_id
        == "L2"
    )
    assert (
        QUADRUPLE_CURRICULUM_V1.level_for_transition(1_000_000).level_id
        == "L4"
    )
    assert (
        QUADRUPLE_CURRICULUM_V1.metadata()["schema_version"]
        == "aiogym.curriculum.v1"
    )


@pytest.mark.parametrize("level", ["L0", "L2", "L4"])
def test_distribution_same_seed_same_episode_spec(level):
    first = QuadrupleTrainingSampler(level=level).sample(101)
    second = QuadrupleTrainingSampler(level=level).sample(101)
    different = QuadrupleTrainingSampler(level=level).sample(102)

    assert first.as_dict() == second.as_dict()
    assert first.resolved_hash == second.resolved_hash
    assert first.resolved_hash != different.resolved_hash
    assert first.difficulty_tags[0] == level


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3", "L4"])
def test_sampled_episode_is_physically_feasible(level):
    sampler = QuadrupleTrainingSampler(level=level)
    for seed in range(20):
        episode = sampler.sample(seed)
        report = validate_quadruple_episode(episode)
        assert report["passed"], report
        assert all(0.0 <= value <= 20.0 for value in episode.initial_state)
        assert episode.reference_schedule[0]["at_step"] == 0
        assert all(
            event["at_step"] < episode.episode_steps
            for event in episode.reference_schedule
        )


def test_parameter_sample_preserves_declared_correlations():
    sampler = QuadrupleTrainingSampler(level="L3")
    episode = sampler.sample(11)
    regime = episode.difficulty_tags[1]
    declaration = sampler.distribution.declaration
    anchor = declaration["plant_distribution"]["anchors"][regime]
    sampled = episode.plant_parameters

    outlet_ratios = np.asarray(sampled["outlet_area"]) / np.asarray(
        anchor["outlet_area"]
    )
    pump_ratios = np.asarray(sampled["pump_gain"]) / np.asarray(
        anchor["pump_gain"]
    )
    gamma_shifts = np.asarray(sampled["gamma"]) - np.asarray(anchor["gamma"])
    assert np.ptp(outlet_ratios) < 1e-12
    assert np.ptp(pump_ratios) < 1e-12
    assert np.ptp(gamma_shifts) < 1e-12


def test_invalid_phase_regime_is_rejected():
    episode = QuadrupleTrainingSampler(level="L2").sample(4)
    invalid = episode.plant_parameters
    invalid["gamma"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="phase margin"):
        validate_quadruple_parameters(
            invalid,
            expected_regime=episode.difficulty_tags[1],
        )


def test_reference_event_times_and_targets_are_not_fixed():
    sampler = QuadrupleTrainingSampler(level="L2")
    schedules = [
        sampler.sample(seed).reference_schedule
        for seed in range(10, 20)
    ]
    event_time_sets = {
        tuple(event["at_step"] for event in schedule[1:])
        for schedule in schedules
    }
    assert len(event_time_sets) > 1

    for episode_seed, schedule in zip(range(10, 20), schedules):
        episode = sampler.sample(episode_seed)
        model = apply_model_params(
            make_model("quadruple"),
            episode.plant_parameters,
        )
        assert all(
            model.is_setpoint_reachable(event["values"])
            for event in schedule
        )


def test_reference_generator_supports_async_feasible_steps_and_dwell():
    distribution = quadruple_training_distribution(
        "L2",
        episode_steps=600,
    )
    specification = distribution.declaration["reference_distribution"]
    model = make_model("quadruple")
    equilibrium = model.initial_state()
    initial_reference = model.controlled_output(equilibrium)
    schedule = QuadrupleReferenceGenerator().generate(
        model,
        initial_reference=initial_reference,
        operating_action=model.default_action(),
        episode_steps=distribution.episode_steps,
        specification=specification,
        rng=np.random.default_rng(42),
        pattern="async_step",
    )
    dwell = specification["minimum_dwell_steps"]
    times = [event["at_step"] for event in schedule]
    assert times[1] >= dwell
    assert all(
        later - earlier >= dwell
        for earlier, later in zip(times[1:], times[2:])
    )
    for previous, event in zip(schedule, schedule[1:]):
        changed = sum(
            abs(float(current) - float(prior)) > 1e-9
            for current, prior in zip(
                event["values"],
                previous["values"],
            )
        )
        assert changed == 1
        assert model.is_setpoint_reachable(event["values"])


@pytest.mark.parametrize(
    "pattern",
    [
        "pulse",
        "ramp",
        "bounded_random_walk",
        "colored_noise",
        "piecewise_regime_shift",
    ],
)
def test_disturbance_generators_respect_declared_bounds(pattern):
    distribution = quadruple_training_distribution("L4")
    specification = distribution.declaration["disturbance_distribution"]
    model = make_model("quadruple")
    schedule = QuadrupleDisturbanceGenerator().generate(
        model,
        episode_steps=distribution.episode_steps,
        specification=specification,
        rng=np.random.default_rng(8),
        force_shift=True,
        pattern=pattern,
    )
    shifted = [event for event in schedule if event["at_step"] > 0]
    assert shifted
    for event in shifted:
        if event["name"] in {
            "pump_flow_factor",
            "outlet_area_factor",
        }:
            assert 0.7 <= event["value"] <= 1.3


def test_paired_shift_changes_only_declared_component():
    nominal, shifted = QuadrupleTrainingSampler(level="L2").sample_paired(
        77,
        shift_component="disturbance",
    )
    nominal_payload = nominal.as_dict()
    shifted_payload = shifted.as_dict()
    for payload in (nominal_payload, shifted_payload):
        payload.pop("episode_spec_id")
        payload.pop("resolved_hash")
    nominal_disturbances = nominal_payload.pop("disturbance_schedule")
    shifted_disturbances = shifted_payload.pop("disturbance_schedule")

    assert nominal_payload == shifted_payload
    assert nominal_disturbances != shifted_disturbances
    assert all(event["at_step"] == 0 for event in nominal_disturbances)
    assert any(event["at_step"] > 0 for event in shifted_disturbances)


def test_curriculum_sampler_selects_level_from_transition_count():
    sampler = QuadrupleCurriculumSampler(episode_steps=300)
    early = sampler.sample(5, transition_count=0)
    late = sampler.sample(5, transition_count=1_000_000)
    assert early.difficulty_tags[0] == "L0"
    assert late.difficulty_tags[0] == "L4"
    assert early.distribution_id != late.distribution_id


def test_generated_episode_executes_through_resolved_reset_path():
    episode = QuadrupleTrainingSampler(
        level="L3",
        episode_steps=300,
    ).sample(19)
    env = make_env(
        "quadruple",
        control_dt=1.0,
        episode_steps=300,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
        noise=False,
        reward_spec="regulation-v1",
        previous_action_obs=True,
        normalize_observations=True,
        tracking_error_obs=True,
    )
    try:
        observation, info = env.reset(options={"episode_spec": episode})
        assert observation.shape == env.observation_space.shape
        assert info["episode_spec_id"] == episode.episode_spec_id
        action = np.full(env.action_space.shape, 0.3, dtype=np.float32)
        for _ in range(5):
            observation, reward, terminated, truncated, info = env.step(
                action
            )
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert not truncated
            if terminated:
                break
        assert info["distribution_hash"] == episode.distribution_hash
    finally:
        env.close()


def test_training_distribution_does_not_embed_fixed_case_profiles():
    declaration = quadruple_training_distribution("L2").declaration
    serialized = repr(declaration)
    assert "case_profile_hash" not in serialized
    assert "fixed-case:" not in serialized
    assert declaration["plant_distribution"]["anchor_source"] == (
        "quadruple parameter profile v1"
    )


def test_quadruple_track_points_to_programmatic_training_distribution():
    track = load_track("quadruple-regulation-generalist-v1")
    assert (
        track.train_distribution_id
        == "quadruple-regulation-training-l2-v1"
    )
    distribution = track.training_distribution()
    assert distribution == load_distribution(track.train_distribution_id)
    assert distribution.scenario_id == track.scenario
    assert distribution.goal == track.goal
    assert distribution.control_dt == track.policy_contract["control_dt"]

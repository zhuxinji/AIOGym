from __future__ import annotations

import json

import numpy as np
import pytest

from aiogym import (
    make_env,
    list_benchmarks,
    list_rewards,
    list_scenarios,
)
from aiogym.core.contracts import ProcessModel


def test_builtin_scenarios_are_listed():
    assert set(list_scenarios()) == {
        "cascade",
        "cstr",
        "crystallization",
        "extraction",
        "heater",
        "hvac",
        "quadruple",
        "three_tank",
    }
    assert set(list_rewards("cstr")) == {"regulation"}
    assert set(list_rewards("cascade")) == {"regulation"}
    assert set(list_rewards("crystallization")) == {"batch-quality"}
    assert set(list_rewards("extraction")) == {"regulation"}
    assert set(list_rewards("heater")) == {"regulation"}
    assert set(list_rewards("hvac")) == {"regulation"}
    assert set(list_rewards("three_tank")) == {"regulation"}
    assert set(list_rewards("quadruple")) == {"regulation"}
    assert set(list_benchmarks("three_tank")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("cascade")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("quadruple")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("hvac")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("cstr")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("crystallization")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("extraction")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("heater")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }


@pytest.mark.parametrize("scenario", list_scenarios())
def test_builtin_models_use_one_process_model_interface(scenario):
    env = make_env(scenario)
    try:
        model = env.unwrapped.model
        assert isinstance(model, ProcessModel)
        assert len(model.outputs(model.initial_state())) == len(model.output_schema())
        assert all(
            {"name", "unit", "low", "high"} <= row.keys()
            for row in model.state_schema()
        )
        assert all(
            {"name", "unit", "low", "high"} <= row.keys()
            for row in model.action_schema()
        )
        assert all(
            {"name", "unit", "low", "high"} <= row.keys()
            for row in model.output_schema()
        )
    finally:
        env.close()


@pytest.mark.parametrize("scenario", list_scenarios())
def test_randomized_validation_cases_are_fixed_by_seed_for_every_scenario(scenario):
    env = make_env(scenario, randomize=True)
    try:
        _, first = env.reset(seed=1_000)
        _, repeated = env.reset(seed=1_000)
        _, different = env.reset(seed=1_001)
    finally:
        env.close()

    assert first["episode_spec"] == repeated["episode_spec"]
    assert first["episode_spec"] != different["episode_spec"]


@pytest.mark.parametrize(
    ("scenario", "expected_observation", "expected_action"),
    [
        ("cascade", (16,), (7,)),
        ("cstr", (6,), (2,)),
        ("crystallization", (8,), (1,)),
        ("extraction", (11,), (2,)),
        ("heater", (5,), (2,)),
        ("hvac", (4,), (2,)),
        ("quadruple", (8,), (2,)),
        ("three_tank", (10,), (4,)),
    ],
)
def test_all_benchmarks_keep_one_scenario_interface(
    scenario,
    expected_observation,
    expected_action,
):
    for benchmark in list_benchmarks(scenario):
        env = make_env(scenario, benchmark=benchmark)
        try:
            observation, info = env.reset(seed=3)
            assert env.observation_space.shape == expected_observation
            assert env.action_space.shape == expected_action
            assert env.observation_space.contains(observation)
            assert info["episode_parameters"] == {"case_seed": 3}
            assert info["benchmark_id"] == benchmark
        finally:
            env.close()


@pytest.mark.parametrize("scenario", list_scenarios())
@pytest.mark.parametrize(
    "benchmark",
    ("tracking", "disturbance-rejection", "boundary-safety"),
)
def test_every_formal_benchmark_resolves_twenty_distinct_reproducible_cases(
    scenario,
    benchmark,
):
    env = make_env(scenario, benchmark=benchmark)
    try:
        base_env = env.unwrapped
        episodes = [
            base_env.benchmark.make_episode(base_env.model, seed) for seed in range(20)
        ]
        serialized = {
            json.dumps(episode.as_dict(), sort_keys=True) for episode in episodes
        }

        assert len(serialized) == 20
        assert base_env.benchmark.make_episode(base_env.model, 0) == episodes[0]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scenario", "horizon"),
    (
        ("cascade", 2100),
        ("crystallization", 50),
        ("cstr", 225),
        ("extraction", 100),
        ("heater", 300),
        ("hvac", 60),
        ("quadruple", 180),
        ("three_tank", 600),
    ),
)
def test_tracking_benchmarks_start_one_target_at_reset(scenario, horizon):
    env = make_env(scenario, benchmark="tracking")
    try:
        env.reset(seed=0)
        episode = env.unwrapped.episode
        initial_output = env.unwrapped.model.outputs(episode.initial_state)

        assert episode.horizon == horizon
        assert not episode.reference_schedule
        assert not np.allclose(initial_output, episode.reference)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scenario", "benchmark", "horizon"),
    (
        ("cascade", "disturbance-rejection", 1000),
        ("cascade", "boundary-safety", 600),
        ("crystallization", "disturbance-rejection", 100),
        ("crystallization", "boundary-safety", 100),
        ("cstr", "disturbance-rejection", 400),
        ("cstr", "boundary-safety", 100),
        ("extraction", "disturbance-rejection", 580),
        ("extraction", "boundary-safety", 100),
        ("heater", "disturbance-rejection", 700),
        ("heater", "boundary-safety", 450),
        ("hvac", "disturbance-rejection", 210),
        ("hvac", "boundary-safety", 60),
        ("quadruple", "disturbance-rejection", 500),
        ("quadruple", "boundary-safety", 180),
        ("three_tank", "disturbance-rejection", 600),
        ("three_tank", "boundary-safety", 600),
    ),
)
def test_safety_benchmark_horizons_and_events(scenario, benchmark, horizon):
    env = make_env(scenario, benchmark=benchmark)
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            spec = info["episode_spec"]
            assert spec["horizon"] == horizon
            assert all(
                int(step) < horizon
                for step in spec["disturbance_schedule"]
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "scenario",
    ("cascade", "cstr", "extraction", "heater", "hvac", "quadruple", "three_tank"),
)
def test_continuous_boundary_cases_are_safe_nonsteady_prerun_states(scenario):
    env = make_env(scenario, benchmark="boundary-safety")
    try:
        for seed in range(20):
            env.reset(seed=seed)
            episode = env.unwrapped.episode
            state = np.asarray(episode.initial_state, dtype=float)
            derivative = np.asarray(
                env.unwrapped.model.dynamics(
                    state,
                    episode.initial_action,
                    episode.disturbances,
                ),
                dtype=float,
            )
            costs = env.unwrapped.model.constraint_costs(
                state, episode.disturbances
            )
            assert not np.allclose(state, env.unwrapped.model.initial_state())
            assert np.max(np.abs(derivative)) > 1e-6
            assert not any(float(value) > 0.0 for value in costs.values())
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scenario", "horizon"),
    (
        ("cascade", 600),
        ("crystallization", 50),
        ("cstr", 225),
        ("extraction", 100),
        ("heater", 300),
        ("hvac", 60),
        ("quadruple", 180),
        ("three_tank", 600),
    ),
)
def test_default_and_randomized_training_match_tracking_timing(scenario, horizon):
    for randomize, seeds in ((False, (0,)), (True, range(20))):
        env = make_env(scenario, randomize=randomize)
        try:
            for seed in seeds:
                _, info = env.reset(seed=seed)
                spec = info["episode_spec"]
                assert spec["horizon"] == horizon
                assert not spec["reference_schedule"]
                if randomize:
                    assert (
                        info["episode_parameters"]["initial_family"]
                        == "interior"
                    )
                initial_output = env.unwrapped.model.outputs(spec["initial_state"])
                assert not np.allclose(initial_output, spec["reference"])
        finally:
            env.close()


@pytest.mark.parametrize(
    "scenario",
    ("cascade", "cstr", "extraction", "heater", "hvac", "quadruple", "three_tank"),
)
def test_continuous_boundary_probability_uses_reproducible_prerun_without_target_shift(
    scenario,
):
    interior = make_env(scenario, randomize=True)
    boundary = make_env(scenario, randomize=True, boundary_probability=1.0)
    try:
        _, interior_info = interior.reset(seed=11)
        _, boundary_info = boundary.reset(seed=11)
    finally:
        interior.close()
        boundary.close()
    assert interior_info["episode_parameters"]["initial_family"] == "interior"
    assert (
        boundary_info["episode_parameters"]["initial_family"]
        == "boundary-prerun"
    )
    assert interior_info["episode_spec"]["reference"] == boundary_info[
        "episode_spec"
    ]["reference"]
    assert not any(
        value > 0.0 for value in boundary_info["constraint_costs"].values()
    )


def test_crystallization_boundary_probability_records_batch_initialization():
    env = make_env(
        "crystallization", randomize=True, boundary_probability=1.0
    )
    try:
        _, info = env.reset(seed=11)
    finally:
        env.close()
    assert info["episode_parameters"]["initial_family"] == "boundary-batch"
    assert not any(value > 0.0 for value in info["constraint_costs"].values())


@pytest.mark.parametrize(
    "scenario",
    (
        "crystallization",
        "cascade",
        "cstr",
        "extraction",
        "heater",
        "hvac",
        "quadruple",
        "three_tank",
    ),
)
def test_training_disturbance_events_fit_unified_tracking_horizon(scenario):
    env = make_env(scenario, randomize=True, disturbance=True)
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            spec = info["episode_spec"]
            schedule = spec["disturbance_schedule"]
            assert schedule
            assert max(int(step) for step in schedule) < spec["horizon"] - int(np.ceil(spec["horizon"] * 0.1))
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scenario", "time_unit"),
    (
        ("cascade", "s"),
        ("crystallization", "s"),
        ("cstr", "s"),
        ("extraction", "h"),
        ("heater", "s"),
        ("hvac", "s"),
        ("quadruple", "s"),
        ("three_tank", "s"),
    ),
)
def test_scenario_models_declare_equation_time_unit(scenario, time_unit):
    env = make_env(scenario)
    try:
        assert env.unwrapped.model.time_unit == time_unit
    finally:
        env.close()


@pytest.mark.parametrize("scenario", list_scenarios())
def test_benchmarks_use_fixed_parameters_and_declared_reward(scenario):
    from aiogym.core.catalog import get_scenario

    definition = get_scenario(scenario)
    expected_parameters = dict(definition.make_model(None).resolved_parameters)
    for benchmark in list_benchmarks(scenario):
        env = make_env(scenario, benchmark=benchmark)
        try:
            assert env.unwrapped.reward.id == definition.benchmarks[benchmark].reward_id
            assert dict(env.unwrapped.model.resolved_parameters) == expected_parameters
        finally:
            env.close()


def test_latest_formal_disturbances_leave_a_recovery_interval():
    import math

    class LatestEventRng:
        def __init__(self):
            self.rng = np.random.default_rng(0)

        def integers(self, low, high=None):
            return (low if high is None else high) - 1

        def __getattr__(self, name):
            return getattr(self.rng, name)

    # Exercise upper bounds, including rare event times absent from seeds 0--19.
    for scenario in list_scenarios():
        env = make_env(scenario, benchmark="disturbance-rejection")
        try:
            episode = env.unwrapped.benchmark.episode_factory(env.unwrapped.model, LatestEventRng())
            end = max(episode.disturbance_schedule)
            assert end < episode.horizon - math.ceil(episode.horizon * 0.1), scenario
            if scenario in {"cstr", "extraction", "quadruple"}:
                assert end <= int(episode.horizon * 0.8), scenario
        finally:
            env.close()

        training_env = make_env(scenario, randomize=True)
        try:
            _, info = training_env.reset(seed=0)
            base = training_env.unwrapped
            schedule = base.scenario.sample_training_disturbance(base.model, LatestEventRng())
            horizon = info["episode_spec"]["horizon"]
            assert max(schedule) < horizon - math.ceil(horizon * 0.1), scenario
        finally:
            training_env.close()

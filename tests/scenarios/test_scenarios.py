from __future__ import annotations

import numpy as np
import pytest

from aiogym import make_env
from aiogym.core.registry import (
    get_reward,
    list_benchmarks,
    list_rewards,
    list_scenarios,
)
from aiogym.scenarios.three_tank.wrappers import Tank3ResidualWrapper


def test_three_tank_rewards_share_one_stable_physical_interface():
    regulation = make_env("three_tank", reward="regulation")
    economic = make_env("three_tank", reward="economic")
    try:
        assert regulation.action_space.shape == economic.action_space.shape == (5,)
        assert regulation.observation_space.shape == economic.observation_space.shape == (
            17,
        )
        assert regulation.model.__class__ is economic.model.__class__
    finally:
        regulation.close()
        economic.close()


def test_only_quadruple_and_three_tank_are_registered():
    assert set(list_scenarios()) == {"quadruple", "three_tank"}
    assert set(list_rewards(scenario="three_tank")) == {
        "regulation",
        "tank3-regulation",
        "economic",
    }
    assert set(list_benchmarks(scenario="three_tank")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks(scenario="quadruple")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert get_reward("three_tank", "economic").primary_metric == (
        "economic_objective"
    )


@pytest.mark.parametrize(
    ("scenario", "expected_observation", "expected_action"),
    [("quadruple", (8,), (2,)), ("three_tank", (17,), (5,))],
)
def test_all_benchmarks_keep_one_scenario_interface(
    scenario,
    expected_observation,
    expected_action,
):
    for benchmark in list_benchmarks(scenario=scenario):
        env = make_env(scenario, benchmark=benchmark)
        try:
            observation, info = env.reset(seed=3)
            assert env.observation_space.shape == expected_observation
            assert env.action_space.shape == expected_action
            assert env.observation_space.contains(observation)
            assert info["episode_parameters"] == {"reset_seed": 3}
            assert info["benchmark_id"] == benchmark
        finally:
            env.close()


def test_quadruple_default_training_episode_has_one_feasible_setpoint_step():
    env = make_env("quadruple")
    try:
        base_env = env.unwrapped
        episode = base_env.default_episode
        target = episode.reference_schedule[300]

        assert base_env.benchmark is None
        assert base_env.episode_family == "default"
        assert episode.horizon == 600
        assert episode.reference == pytest.approx(
            base_env.model.default_setpoint_vector()
        )
        assert dict(episode.reference_schedule) == pytest.approx(
            {300: (15.0, 10.5)}
        )

        target_action = base_env.model.tracking_steady_state_action(target)
        assert target_action is not None
        assert np.all(np.asarray(target_action) > 0.0)
        assert np.all(np.asarray(target_action) < 1.0)
    finally:
        env.close()


def test_quadruple_smooth_regulation_penalizes_action_moves():
    env = make_env("quadruple", reward="smooth-regulation")
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(
            np.asarray((0.5, 0.5), dtype=np.float32)
        )
    finally:
        env.close()
    assert not terminated
    assert not truncated
    assert info["reward_terms"]["slew"] == pytest.approx(-0.8)
    assert reward == pytest.approx(sum(info["reward_terms"].values()))


def test_three_tank_default_training_episode_has_one_feasible_heating_step():
    env = make_env("three_tank")
    try:
        base_env = env.unwrapped
        episode = base_env.default_episode
        target = episode.reference_schedule[600]

        assert base_env.benchmark is None
        assert base_env.episode_family == "default"
        assert episode.horizon == 3000
        assert tuple(episode.reference[:3]) == pytest.approx((0.18,) * 3)
        assert tuple(target[:3]) == pytest.approx((0.30,) * 3)
        assert episode.reference[5] == pytest.approx(23.2193557502275)
        assert target[5] == pytest.approx(23.7)

        start_action = base_env.model.tracking_steady_state_action(
            episode.reference
        )
        target_action = base_env.model.tracking_steady_state_action(target)
        assert start_action is not None
        assert target_action is not None
        assert np.all(np.asarray(start_action) > 0.0)
        assert np.all(np.asarray(start_action) < 1.0)
        assert np.all(np.asarray(target_action) > 0.0)
        assert np.all(np.asarray(target_action) < 1.0)
        assert target_action[4] > start_action[4]
    finally:
        env.close()


def test_three_tank_regulation_uses_thermal_working_scale():
    env = make_env("three_tank")
    try:
        env.reset(seed=0)
        model = env.unwrapped.model
        action = np.asarray(model.tracking_steady_state_action(env.unwrapped.y_sp))
        _, reward, terminated, truncated, info = env.step(action)

        assert not terminated
        assert not truncated
        assert reward == pytest.approx(sum(info["reward_terms"].values()))
        assert abs(reward) < 1e-6

        equilibrium_state = np.asarray(env.unwrapped.default_episode.initial_state)
        heated_state = equilibrium_state.copy()
        heated_state[1] += 10.0
        reference = np.asarray(env.unwrapped.default_episode.reference)
        target_action = np.asarray(model.tracking_steady_state_action(reference))
        context = {
            "model": model,
            "reference": reference,
            "control_dt": 1.0,
            "disturbances": model.default_disturbances(),
        }
        one_channel_error, _ = env.unwrapped.reward.function(
            equilibrium_state,
            target_action,
            heated_state,
            context,
        )
        assert one_channel_error == pytest.approx(-1.0 / 6.0)

        shifted_action = target_action.copy()
        shifted_action[0] += 0.1
        action_penalty, terms = env.unwrapped.reward.function(
            equilibrium_state,
            shifted_action,
            equilibrium_state,
            context,
        )
        assert terms["tracking_error"] == pytest.approx(0.0)
        assert terms["feedforward"] == pytest.approx(-0.001)
        assert action_penalty == pytest.approx(-0.001)
    finally:
        env.close()


def test_tracking_benchmarks_cover_a_visible_fraction_of_the_output_scale():
    expected = {
        "quadruple": {
            "horizon": 600,
            "schedule": {
                120: (16.0, 10.0),
                360: (10.0, 16.0),
            },
            "minimum_span_fraction": 0.30,
        },
        "three_tank": {
            "horizon": 4200,
            "schedule_step": 900,
            "initial_levels": (0.12, 0.12, 0.12),
            "target_levels": (0.30, 0.30, 0.30),
            "initial_tank3_temperature": 20.5,
            "target_tank3_temperature": 23.0,
            "minimum_span_fraction": 0.45,
        },
    }
    for scenario, contract in expected.items():
        env = make_env(scenario, benchmark="tracking")
        try:
            episode = env.unwrapped.default_episode
            assert episode.horizon == contract["horizon"]
            if scenario == "quadruple":
                assert dict(episode.reference_schedule) == pytest.approx(
                    contract["schedule"]
                )
                values = np.asarray(
                    [episode.reference, *episode.reference_schedule.values()]
                )
                span = np.ptp(values, axis=0)
            else:
                assert tuple(episode.reference[:3]) == pytest.approx(
                    contract["initial_levels"]
                )
                assert set(episode.reference_schedule) == {
                    contract["schedule_step"]
                }
                target = episode.reference_schedule[contract["schedule_step"]]
                assert tuple(target[:3]) == pytest.approx(
                    contract["target_levels"]
                )
                assert episode.reference[5] == pytest.approx(
                    contract["initial_tank3_temperature"]
                )
                assert target[5] == pytest.approx(
                    contract["target_tank3_temperature"]
                )
                assert target[5] > episode.reference[5]
                start_action = env.unwrapped.model.tracking_steady_state_action(
                    episode.reference,
                    episode.disturbances,
                )
                target_action = env.unwrapped.model.tracking_steady_state_action(
                    target,
                    episode.disturbances,
                )
                assert start_action is not None
                assert target_action is not None
                assert target_action[0] == pytest.approx(start_action[0])
                assert target_action[4] > start_action[4]
                span = np.abs(
                    np.asarray(target[:3]) - np.asarray(episode.reference[:3])
                )
            high = np.asarray(
                [row["high"] for row in env.unwrapped.model.output_schema()]
            )
            assert np.all(
                span / high[: span.size]
                >= contract["minimum_span_fraction"] - 1e-12
            )
        finally:
            env.close()


def test_three_tank_tracking_ranks_safety_before_error_integral():
    env = make_env("three_tank", benchmark="tracking")
    try:
        assert env.unwrapped.benchmark.ranking_metrics == (
            ("unsafe_rate", "minimize"),
            ("tracking_iae", "minimize"),
        )
    finally:
        env.close()


def test_three_tank_residual_control_is_an_explicit_wrapper():
    base = make_env(
        "three_tank",
        benchmark="tracking",
    )
    wrapped = Tank3ResidualWrapper(base)
    try:
        observation, _ = wrapped.reset(seed=3)
        assert base.action_space.shape == (5,)
        assert wrapped.action_space.shape == (2,)
        assert observation.shape == (17,)
        _, reward, terminated, _, info = wrapped.step([0.0, 0.0])
        assert np.isfinite(reward)
        assert not terminated
        assert info["commanded_action"].shape == (2,)
        assert info["resolved_physical_action"].shape == (5,)
        assert info["applied_action"].shape == (5,)
        assert base.action_space.contains(info["resolved_physical_action"])
    finally:
        wrapped.close()


@pytest.mark.parametrize("scenario", ("quadruple", "three_tank"))
def test_benchmarks_use_fixed_scenario_defaults(scenario):
    from aiogym.core.registry import get_scenario

    definition = get_scenario(scenario)
    expected_parameters = dict(definition.make_model(None).resolved_parameters)
    for benchmark in list_benchmarks(scenario=scenario):
        env = make_env(scenario, benchmark=benchmark)
        try:
            assert env.unwrapped.reward.id == definition.default_reward
            assert dict(env.unwrapped.model.resolved_parameters) == expected_parameters
        finally:
            env.close()


@pytest.mark.parametrize(
    ("scenario", "parameters", "action"),
    (
        ("quadruple", {"pump_gain": [3.0, 3.1]}, [0.8, 0.2]),
        ("three_tank", {"heater_power": 1800.0}, [0.5, 0.5, 0.5, 0.5, 1.0]),
    ),
)
def test_parameter_override_is_applied_before_dynamics_without_changing_interface(
    scenario,
    parameters,
    action,
):
    base = make_env(scenario)
    changed = make_env(scenario, parameters=parameters)
    try:
        assert changed.observation_space.shape == base.observation_space.shape
        assert changed.action_space.shape == base.action_space.shape
        state = base.model.initial_state()
        disturbances = base.model.default_disturbances()
        base_derivative = base.model.dynamics(state, action, disturbances)
        changed_derivative = changed.model.dynamics(state, action, disturbances)
        assert not np.allclose(base_derivative, changed_derivative)
        for name, value in parameters.items():
            expected = tuple(value) if isinstance(value, list) else value
            assert changed.model.resolved_parameters[name] == expected
        with pytest.raises(TypeError):
            changed.model.resolved_parameters["new"] = 1.0
    finally:
        base.close()
        changed.close()


@pytest.mark.parametrize("scenario", ("quadruple", "three_tank"))
def test_parameter_override_rejects_unknown_and_nonfinite_values(scenario):
    with pytest.raises(ValueError, match="unknown .* parameters"):
        make_env(scenario, parameters={"not_a_parameter": 1.0})
    with pytest.raises(ValueError, match="finite"):
        name = "gravity" if scenario == "quadruple" else "heater_power"
        make_env(scenario, parameters={name: float("nan")})


def test_parameter_override_rejects_physically_invalid_values():
    with pytest.raises(ValueError, match="gamma"):
        make_env("quadruple", parameters={"gamma": [1.2, 0.6]})
    with pytest.raises(ValueError, match="level parameters"):
        make_env("three_tank", parameters={"height_max": [0.3, 0.3, 0.3]})


@pytest.mark.parametrize(
    "legacy",
    ("cascade/regulation", "cascade/economic", "cascade_recirculating/regulation"),
)
def test_removed_legacy_scenario_aliases_are_not_registered(legacy):
    with pytest.raises(KeyError, match="unknown scenario"):
        make_env(legacy)

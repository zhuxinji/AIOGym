from __future__ import annotations

import numpy as np
import pytest

from aiogym import (
    make_env,
    list_benchmarks,
    list_rewards,
    list_scenarios,
)


def test_only_quadruple_and_three_tank_are_registered():
    assert set(list_scenarios()) == {"quadruple", "three_tank"}
    assert set(list_rewards("three_tank")) == {
        "regulation",
        "thermal_regulation",
    }
    assert set(list_rewards("quadruple")) == {"regulation"}
    assert set(list_benchmarks("three_tank")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    assert set(list_benchmarks("quadruple")) == {
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    }
    with pytest.raises(KeyError, match="unknown reward 'economic'"):
        make_env("three_tank", reward="economic")
@pytest.mark.parametrize(
    ("scenario", "expected_observation", "expected_action"),
    [("quadruple", (6,), (2,)), ("three_tank", (13,), (5,))],
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


def test_quadruple_default_training_episode_starts_with_large_feasible_error():
    env = make_env("quadruple")
    try:
        base_env = env.unwrapped
        episode = base_env.default_episode
        target = episode.reference_schedule[300]
        initial_output = np.asarray(base_env.model.outputs(episode.initial_state))
        initial_reference = np.asarray(episode.reference)

        assert base_env.benchmark is None
        assert base_env.episode_family == "default"
        assert episode.horizon == 600
        assert episode.reference == pytest.approx((18.0, 8.0))
        assert dict(episode.reference_schedule) == pytest.approx(
            {300: (8.0, 18.0)}
        )
        assert np.all(np.abs(initial_reference - initial_output) >= 4.0)

        initial_action = base_env.model.tracking_steady_state_action(
            initial_reference
        )
        target_action = base_env.model.tracking_steady_state_action(target)
        assert initial_action is not None
        assert target_action is not None
        assert np.all(np.asarray(initial_action) > 0.0)
        assert np.all(np.asarray(initial_action) < 1.0)
        assert np.all(np.asarray(target_action) > 0.0)
        assert np.all(np.asarray(target_action) < 1.0)
    finally:
        env.close()


def test_quadruple_observation_contains_normalized_setpoint_not_error():
    env = make_env("quadruple")
    try:
        observation, _ = env.reset(seed=0)
        initial_reference = np.asarray(env.unwrapped.default_episode.reference)
        assert observation[-2:] == pytest.approx(initial_reference / 20.0)

        action = np.asarray(env.unwrapped.model.default_action(), dtype=np.float32)
        for _ in range(300):
            observation, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated
        assert observation[-2:] == pytest.approx(np.asarray((8.0, 18.0)) / 20.0)
    finally:
        env.close()


def test_three_tank_default_training_episode_has_staged_feasible_targets():
    env = make_env("three_tank")
    try:
        base_env = env.unwrapped
        episode = base_env.default_episode
        level_target = episode.reference_schedule[600]

        assert base_env.benchmark is None
        assert base_env.episode_family == "default"
        assert episode.horizon == 2400
        assert tuple(episode.reference_schedule) == (600,)
        assert episode.reference == pytest.approx(
            base_env.model.outputs(episode.initial_state)
        )
        assert episode.reference == pytest.approx(
            (0.225, 24.0, 0.225, 23.3582029683, 0.225, 22.8193817941)
        )
        assert level_target[0::2] == pytest.approx((0.225, 0.225, 0.375))
        assert level_target[1::2] == pytest.approx(episode.reference[1::2])

        start_action = base_env.model.tracking_steady_state_action(
            episode.reference
        )
        level_action = base_env.model.tracking_steady_state_action(level_target)
        assert start_action is not None
        assert level_action is not None
        for action in (start_action, level_action):
            assert np.all(np.asarray(action) > 0.0)
            assert np.all(np.asarray(action) < 1.0)
        assert level_action[4] == pytest.approx(start_action[4])
    finally:
        env.close()


def test_three_tank_regulation_tracks_only_levels():
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
        heated_state[5] += 3.0
        reference = np.asarray(env.unwrapped.default_episode.reference)
        target_action = np.asarray(model.tracking_steady_state_action(reference))
        context = {
            "model": model,
            "reference": reference,
            "control_dt": 1.0,
            "disturbances": model.default_disturbances(),
            "episode": env.unwrapped.default_episode,
            "step_index": 0,
            "constraint_costs": {},
        }
        one_channel_error, _ = env.unwrapped.reward.function(
            equilibrium_state,
            target_action,
            heated_state,
            context,
        )
        assert one_channel_error == pytest.approx(0.0)

        raised_level_state = equilibrium_state.copy()
        raised_level_state[4] += 0.1
        one_level_error, _ = env.unwrapped.reward.function(
            equilibrium_state,
            target_action,
            raised_level_state,
            context,
        )
        assert one_level_error == pytest.approx(-1.0 / 3.0)

        shifted_action = target_action.copy()
        shifted_action[0] += 0.1
        action_penalty, terms = env.unwrapped.reward.function(
            equilibrium_state,
            shifted_action,
            equilibrium_state,
            context,
        )
        assert terms["tracking_error"] == pytest.approx(0.0)
        assert terms["early_termination"] == pytest.approx(0.0)
        assert "feedforward" not in terms
        assert action_penalty == pytest.approx(0.0)

        unsafe_context = {
            **context,
            "step_index": 670,
            "constraint_costs": {"negative_level": 1.0},
        }
        unsafe_reward, unsafe_terms = env.unwrapped.reward.function(
            equilibrium_state,
            target_action,
            equilibrium_state,
            unsafe_context,
        )
        assert unsafe_terms["early_termination"] == pytest.approx(-3458.0)
        assert unsafe_reward == pytest.approx(-3458.0)
    finally:
        env.close()


def test_three_tank_thermal_regulation_still_tracks_temperature():
    env = make_env("three_tank", reward="thermal_regulation")
    try:
        env.reset(seed=0)
        base_env = env.unwrapped
        equilibrium_state = np.asarray(base_env.default_episode.initial_state)
        heated_state = equilibrium_state.copy()
        heated_state[5] += 3.0
        reference = np.asarray(base_env.default_episode.reference)
        context = {
            "model": base_env.model,
            "reference": reference,
            "control_dt": 1.0,
            "episode": base_env.default_episode,
            "step_index": 0,
            "constraint_costs": {},
        }
        reward, _ = base_env.reward.function(
            equilibrium_state,
            base_env.model.default_action(),
            heated_state,
            context,
        )
        assert reward == pytest.approx(-1.0 / 6.0)
    finally:
        env.close()


def test_quadruple_tracking_case_seeds_resolve_feasible_staged_equilibria():
    env = make_env("quadruple", benchmark="tracking")
    specs = []
    try:
        for seed in range(10):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            references = (
                episode.reference,
                episode.reference_schedule[120],
                episode.reference_schedule[360],
            )
            reference_array = np.asarray(references)

            assert episode.horizon == 600
            assert tuple(episode.reference_schedule) == (120, 360)
            assert np.all((7.0 <= reference_array) & (reference_array <= 16.0))
            assert np.all(np.abs(reference_array[1] - reference_array[0]) >= 3.0)
            assert np.all(np.abs(reference_array[2] - reference_array[1]) >= 3.0)

            actions = [
                env.unwrapped.model.tracking_steady_state_action(reference)
                for reference in references
            ]
            states = [
                env.unwrapped.model.tracking_steady_state_state(reference)
                for reference in references
            ]
            assert all(action is not None for action in actions)
            assert all(state is not None for state in states)
            assert episode.initial_action == pytest.approx(actions[0])
            assert episode.initial_state == pytest.approx(states[0])
            assert all(
                np.all((0.02 <= np.asarray(action)) & (np.asarray(action) <= 0.95))
                for action in actions
            )
            assert all(
                np.all((0.0 <= np.asarray(state)) & (np.asarray(state) <= 20.0))
                for state in states
            )
            for state, action in zip(states, actions):
                derivative = env.unwrapped.model.dynamics(
                    state,
                    action,
                    episode.disturbances,
                )
                assert max(abs(float(value)) for value in derivative) < 1e-10
    finally:
        env.close()
    assert len({repr(spec) for spec in specs}) == len(specs)


def test_three_tank_tracking_case_seeds_resolve_feasible_staged_equilibria():
    env = make_env("three_tank", benchmark="tracking")
    specs = []
    try:
        for seed in range(10):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            level_target = episode.reference_schedule[600]
            start_levels = np.asarray(episode.reference[0::2])
            target_levels = np.asarray(level_target[0::2])

            assert episode.horizon == 2400
            assert tuple(episode.reference_schedule) == (600,)
            assert np.all((0.125 <= start_levels) & (start_levels <= 0.4))
            assert np.all((0.125 <= target_levels) & (target_levels <= 0.4))
            assert np.all(np.abs(target_levels - start_levels) >= 0.05)
            assert level_target[1::2] == pytest.approx(episode.reference[1::2])
            assert 20.5 <= episode.reference[5] <= 26.0

            actions = [
                env.unwrapped.model.tracking_steady_state_action(reference)
                for reference in (episode.reference, level_target)
            ]
            assert all(action is not None for action in actions)
            assert episode.initial_action == pytest.approx(actions[0])
            assert all(
                np.all((0.02 <= np.asarray(action)) & (np.asarray(action) <= 0.85))
                for action in actions
            )
            assert actions[1][0] == pytest.approx(actions[0][0])
            flow_lpm = (
                env.unwrapped.model.step_info(
                    episode.initial_state,
                    episode.initial_action,
                    episode.disturbances,
                )["P101_flow_m3s"]
                * 60000.0
            )
            assert 1.0 <= flow_lpm <= 6.0
            derivative = env.unwrapped.model.dynamics(
                episode.initial_state,
                episode.initial_action,
                episode.disturbances,
            )
            assert max(abs(float(value)) for value in derivative) < 1e-10
    finally:
        env.close()
    assert len({repr(spec) for spec in specs}) == len(specs)


def test_three_tank_benchmarks_rank_safety_before_return():
    for benchmark in list_benchmarks("three_tank"):
        env = make_env("three_tank", benchmark=benchmark)
        try:
            assert env.unwrapped.benchmark.ranking_metrics == (
                ("unsafe_rate", "minimize"),
                ("return", "maximize"),
            )
        finally:
            env.close()


def test_quadruple_benchmarks_rank_safety_before_return():
    for benchmark in list_benchmarks("quadruple"):
        env = make_env("quadruple", benchmark=benchmark)
        try:
            assert env.unwrapped.benchmark.ranking_metrics == (
                ("unsafe_rate", "minimize"),
                ("return", "maximize"),
            )
        finally:
            env.close()


@pytest.mark.parametrize("scenario", ("quadruple", "three_tank"))
def test_benchmarks_use_fixed_parameters_and_declared_reward(scenario):
    from aiogym.core.registry import get_scenario

    definition = get_scenario(scenario)
    expected_parameters = dict(definition.make_model(None).resolved_parameters)
    for benchmark in list_benchmarks(scenario):
        env = make_env(scenario, benchmark=benchmark)
        try:
            assert env.unwrapped.reward.id == definition.benchmarks[benchmark].reward_id
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

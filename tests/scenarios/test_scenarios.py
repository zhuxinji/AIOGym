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


def test_builtin_scenarios_are_registered():
    assert set(list_scenarios()) == {
        "cstr",
        "crystallization",
        "extraction",
        "heater",
        "hvac",
        "quadruple",
        "three_tank",
    }
    assert set(list_rewards("cstr")) == {"regulation"}
    assert set(list_rewards("crystallization")) == {"regulation"}
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


@pytest.mark.parametrize(
    ("scenario", "expected_observation", "expected_action"),
    [
        ("cstr", (4,), (2,)),
        ("crystallization", (7,), (1,)),
        ("extraction", (11,), (2,)),
        ("heater", (5,), (2,)),
        ("hvac", (4,), (2,)),
        ("quadruple", (6,), (2,)),
        ("three_tank", (6,), (4,)),
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
        ("three_tank", "disturbance-rejection", 1800),
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
    ("scenario", "horizon"),
    (
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
                initial_output = env.unwrapped.model.outputs(spec["initial_state"])
                assert not np.allclose(initial_output, spec["reference"])
        finally:
            env.close()


@pytest.mark.parametrize(
    "scenario",
    (
        "crystallization",
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
            assert max(int(step) for step in schedule) < spec["horizon"]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scenario", "time_unit"),
    (
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


def test_quadruple_default_training_episode_tracks_one_feasible_target_at_reset():
    env = make_env("quadruple")
    try:
        base_env = env.unwrapped
        episode = base_env.default_episode
        initial_output = np.asarray(base_env.model.outputs(episode.initial_state))
        target = np.asarray(episode.reference)

        assert base_env.benchmark is None
        assert base_env.episode_family == "default"
        assert episode.horizon == 180
        assert episode.reference == pytest.approx((8.0, 18.0))
        assert not episode.reference_schedule
        assert np.all(np.abs(target - initial_output) >= 4.0)

        target_action = base_env.model.tracking_steady_state_action(target)
        assert target_action is not None
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
        observation, _, terminated, truncated, _ = env.step(action)
        assert not terminated
        assert not truncated
        assert observation[-2:] == pytest.approx(initial_reference / 20.0)
    finally:
        env.close()


def test_three_tank_default_training_episode_tracks_benchmark_seed_zero_at_reset():
    env = make_env("three_tank")
    benchmark_env = make_env("three_tank", benchmark="tracking")
    try:
        base_env = env.unwrapped
        episode = base_env.default_episode
        _, benchmark_info = benchmark_env.reset(seed=0)
        initial_output = np.asarray(base_env.model.outputs(episode.initial_state))

        assert base_env.benchmark is None
        assert base_env.episode_family == "default"
        assert episode.horizon == 600
        assert not episode.reference_schedule
        assert episode == benchmark_env.unwrapped.episode
        assert benchmark_info["episode_parameters"] == {"case_seed": 0}

        start_action = base_env.model.tracking_steady_state_action(initial_output)
        target_action = base_env.model.tracking_steady_state_action(episode.reference)
        assert start_action is not None
        assert target_action is not None
        for action in (start_action, target_action):
            assert np.all(np.asarray(action) > 0.0)
            assert np.all(np.asarray(action) < 1.0)
        assert np.all(np.abs(np.asarray(episode.reference) - initial_output) >= 0.05)
    finally:
        env.close()
        benchmark_env.close()


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
        assert info["reward_terms"]["tracking_error"] < 0.0

        equilibrium_state = np.asarray(env.unwrapped.default_episode.initial_state)
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
        target_state = np.asarray(
            model.tracking_steady_state_state(reference), dtype=float
        )
        raised_level_state = target_state.copy()
        raised_level_state[2] += 0.1
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
            target_state,
            context,
        )
        assert terms["tracking_error"] == pytest.approx(0.0)
        assert terms["early_termination"] == pytest.approx(0.0)
        assert "feedforward" not in terms
        assert action_penalty == pytest.approx(0.0)

        unsafe_context = {
            **context,
            "step_index": 70,
            "constraint_costs": {"negative_level": 1.0},
        }
        unsafe_reward, unsafe_terms = env.unwrapped.reward.function(
            equilibrium_state,
            target_action,
            target_state,
            unsafe_context,
        )
        assert unsafe_terms["early_termination"] == pytest.approx(-1058.0)
        assert unsafe_reward == pytest.approx(-1058.0)
    finally:
        env.close()


def test_quadruple_tracking_cases_start_tracking_one_feasible_target():
    env = make_env("quadruple", benchmark="tracking")
    specs = []
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            start_reference = env.unwrapped.model.outputs(episode.initial_state)
            references = (start_reference, episode.reference)
            reference_array = np.asarray(references)

            assert episode.horizon == 180
            assert not episode.reference_schedule
            assert np.all((7.0 <= reference_array) & (reference_array <= 16.0))
            assert np.all(np.abs(reference_array[1] - reference_array[0]) >= 3.0)

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


def test_three_tank_tracking_cases_start_tracking_one_feasible_target():
    env = make_env("three_tank", benchmark="tracking")
    specs = []
    moves = []
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            start_reference = env.unwrapped.model.outputs(episode.initial_state)
            references = [start_reference, episode.reference]
            level_sets = [np.asarray(reference) for reference in references]

            assert episode.horizon == 600
            assert not episode.reference_schedule
            assert all(
                np.all((0.125 <= levels) & (levels <= 0.4))
                for levels in level_sets
            )
            move = np.abs(level_sets[1] - level_sets[0])
            assert np.all(move >= 0.05)
            moves.extend(move.tolist())

            actions = [
                env.unwrapped.model.tracking_steady_state_action(reference)
                for reference in references
            ]
            assert all(action is not None for action in actions)
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
    assert max(moves) > 0.12


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
        (
            "three_tank",
            {"pump_flow_max": 20.0 / 60000.0},
            [0.5, 0.5, 0.5, 0.5],
        ),
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
        name = "gravity" if scenario == "quadruple" else "pump_flow_max"
        make_env(scenario, parameters={name: float("nan")})


def test_parameter_override_rejects_physically_invalid_values():
    with pytest.raises(ValueError, match="gamma"):
        make_env("quadruple", parameters={"gamma": [1.2, 0.6]})
    with pytest.raises(ValueError, match="level parameters"):
        make_env("three_tank", parameters={"height_max": [0.3, 0.3, 0.3]})

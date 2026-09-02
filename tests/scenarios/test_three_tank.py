from __future__ import annotations

import numpy as np
import pytest

from aiogym import list_benchmarks, make_env


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
            "previous_applied_action": target_action.copy(),
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
        assert terms["slew"] == pytest.approx(-0.0025)
        assert "feedforward" not in terms
        assert action_penalty == pytest.approx(-0.0025)

        unchanged_action_reward, unchanged_terms = env.unwrapped.reward.function(
            equilibrium_state,
            target_action,
            target_state,
            context,
        )
        assert unchanged_terms["slew"] == pytest.approx(0.0)
        assert unchanged_action_reward == pytest.approx(0.0)

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
            assert 3.0 <= flow_lpm <= 8.0
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


def test_three_tank_parameter_overrides_are_strict_and_change_dynamics():
    parameters = {"pump_flow_max": 20.0 / 60000.0}
    base = make_env("three_tank")
    changed = make_env("three_tank", parameters=parameters)
    try:
        assert changed.observation_space.shape == base.observation_space.shape
        assert changed.action_space.shape == base.action_space.shape
        state = base.model.initial_state()
        action = [0.5, 0.5, 0.5, 0.5]
        disturbances = base.model.default_disturbances()
        assert not np.allclose(
            base.model.dynamics(state, action, disturbances),
            changed.model.dynamics(state, action, disturbances),
        )
        assert changed.model.resolved_parameters["pump_flow_max"] == pytest.approx(
            parameters["pump_flow_max"]
        )
        with pytest.raises(TypeError):
            changed.model.resolved_parameters["new"] = 1.0
    finally:
        base.close()
        changed.close()

    with pytest.raises(ValueError, match="unknown .* parameters"):
        make_env("three_tank", parameters={"not_a_parameter": 1.0})
    with pytest.raises(ValueError, match="finite"):
        make_env("three_tank", parameters={"pump_flow_max": float("nan")})
    with pytest.raises(ValueError, match="level parameters"):
        make_env("three_tank", parameters={"height_max": [0.3, 0.3, 0.3]})

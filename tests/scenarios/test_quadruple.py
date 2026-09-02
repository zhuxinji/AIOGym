from __future__ import annotations

import numpy as np
import pytest

from aiogym import list_benchmarks, make_env


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


def test_quadruple_parameter_overrides_are_strict_and_change_dynamics():
    base = make_env("quadruple")
    changed = make_env("quadruple", parameters={"pump_gain": [3.0, 3.1]})
    try:
        assert changed.observation_space.shape == base.observation_space.shape
        assert changed.action_space.shape == base.action_space.shape
        state = base.model.initial_state()
        action = [0.8, 0.2]
        disturbances = base.model.default_disturbances()
        assert not np.allclose(
            base.model.dynamics(state, action, disturbances),
            changed.model.dynamics(state, action, disturbances),
        )
        assert changed.model.resolved_parameters["pump_gain"] == (3.0, 3.1)
        with pytest.raises(TypeError):
            changed.model.resolved_parameters["new"] = 1.0
    finally:
        base.close()
        changed.close()

    with pytest.raises(ValueError, match="unknown .* parameters"):
        make_env("quadruple", parameters={"not_a_parameter": 1.0})
    with pytest.raises(ValueError, match="finite"):
        make_env("quadruple", parameters={"gravity": float("nan")})
    with pytest.raises(ValueError, match="gamma"):
        make_env("quadruple", parameters={"gamma": [1.2, 0.6]})

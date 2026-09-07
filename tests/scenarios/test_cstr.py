from __future__ import annotations

import numpy as np
import pytest

import aiogym


def test_cstr_default_environment_starts_at_an_exact_equilibrium():
    env = aiogym.make_env("cstr")
    try:
        observation, info = env.reset(seed=3)
        model = env.unwrapped.model
        state = model.initial_state()
        action = model.default_action()
        derivative = model.dynamics(state, action, model.default_disturbances())

        assert env.observation_space.shape == (6,)
        assert env.action_space.shape == (2,)
        assert env.observation_space.contains(observation)
        assert state == pytest.approx([0.11771023406521823, 60.0])
        assert action == pytest.approx([0.5, 0.2634990876486953])
        assert derivative == pytest.approx([0.0, 0.0], abs=1e-12)
        assert info["y"] == pytest.approx(state)
        assert [row["name"] for row in model.output_schema()] == [
            "reactant_concentration",
            "reactor_temperature",
        ]
        assert observation[2:4] == pytest.approx(
            [
                (0.075 - 0.02) / 0.18,
                (72.0 - 45.0) / 45.0,
            ]
        )
        assert observation[4:] == pytest.approx(
            [
                (state[0] - 0.075) / 0.18,
                (state[1] - 72.0) / 45.0,
            ]
        )
        assert [row["name"] for row in model.observation_schema()] == [
            "reactant_concentration",
            "reactor_temperature",
            "reactant_concentration_setpoint",
            "reactor_temperature_setpoint",
            "reactant_concentration_tracking_error",
            "reactor_temperature_tracking_error",
        ]
        assert [row["kind"] for row in model.observation_schema()] == [
            "measurement",
            "measurement",
            "reference",
            "reference",
            "derived",
            "derived",
        ]
    finally:
        env.close()


def test_cstr_tracking_error_observation_is_scaled_and_clipped():
    env = aiogym.make_env("cstr")
    try:
        model = env.unwrapped.model
        reference = [0.10, 60.0]
        observation = model.observation(
            [0.118, 64.5],
            reference,
            model.default_action(),
            model.default_disturbances(),
        )
        assert observation[4:] == pytest.approx([0.1, 0.1])

        clipped = model.observation(
            [1.5, 0.0],
            reference,
            model.default_action(),
            model.default_disturbances(),
        )
        assert clipped[4:] == pytest.approx([1.0, -1.0])
    finally:
        env.close()


def test_cstr_requires_both_action_channels():
    env = aiogym.make_env("cstr")
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="action"):
            env.step(np.asarray([0.5], dtype=np.float32))
        _, _, terminated, truncated, info = env.step(
            np.asarray([0.45, 0.30], dtype=np.float32)
        )
        assert not terminated
        assert not truncated
        assert info["applied_action"] == pytest.approx([0.45, 0.30])
    finally:
        env.close()


def test_cstr_parameter_overrides_are_strict_and_change_the_dynamics():
    base = aiogym.make_env("cstr")
    changed = aiogym.make_env(
        "cstr",
        parameters={"maximum_dilution_rate": 0.03},
    )
    try:
        state = [0.2, 65.0]
        action = [0.7, 0.4]
        disturbances = base.unwrapped.model.default_disturbances()
        assert not np.allclose(
            base.unwrapped.model.dynamics(state, action, disturbances),
            changed.unwrapped.model.dynamics(state, action, disturbances),
        )
        assert changed.unwrapped.model.resolved_parameters[
            "maximum_dilution_rate"
        ] == pytest.approx(0.03)
    finally:
        base.close()
        changed.close()

    with pytest.raises(ValueError, match="unknown cstr parameters"):
        aiogym.make_env("cstr", parameters={"unknown": 1.0})
    with pytest.raises(ValueError, match="finite"):
        aiogym.make_env(
            "cstr",
            parameters={"cooling_coefficient": float("nan")},
        )
    with pytest.raises(ValueError, match="positive"):
        aiogym.make_env(
            "cstr",
            parameters={"maximum_dilution_rate": 0.0},
        )


def test_cstr_regulation_scores_concentration_and_temperature_equally():
    env = aiogym.make_env("cstr")
    try:
        model = env.unwrapped.model
        state = np.asarray(model.initial_state(), dtype=float)
        reference = np.asarray(model.default_setpoint_vector(), dtype=float)
        context = {
            "model": model,
            "reference": reference,
            "control_dt": env.unwrapped.control_dt,
        }
        concentration_shift = state.copy()
        concentration_shift[0] += 0.18
        temperature_shift = state.copy()
        temperature_shift[1] += 45.0
        concentration_reward, _ = env.unwrapped.reward.function(
            state,
            model.default_action(),
            concentration_shift,
            context,
        )
        temperature_reward, _ = env.unwrapped.reward.function(
            state,
            model.default_action(),
            temperature_shift,
            context,
        )
        assert concentration_reward == pytest.approx(-0.5)
        assert temperature_reward == pytest.approx(-0.5)
    finally:
        env.close()


def test_cstr_tracking_cases_start_tracking_one_feasible_target():
    env = aiogym.make_env("cstr", benchmark="tracking")
    specs = []
    try:
        model = env.unwrapped.model
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            start_reference = model.outputs(episode.initial_state)
            references = (start_reference, episode.reference)
            values = np.asarray(references, dtype=float)
            assert episode.horizon == 225
            assert not episode.reference_schedule
            assert np.all((0.02 <= values[:, 0]) & (values[:, 0] <= 0.20))
            assert np.all((50.0 <= values[:, 1]) & (values[:, 1] <= 82.0))
            assert np.all(np.abs(values[1] - values[0]) >= [0.02, 5.0])
            assert episode.initial_state == pytest.approx(
                model.tracking_steady_state_state(references[0])
            )
            assert episode.initial_action == pytest.approx(
                model.tracking_steady_state_action(references[0])
            )
            for reference in references:
                action = model.tracking_steady_state_action(reference)
                state = model.tracking_steady_state_state(reference)
                assert action is not None
                assert state is not None
                assert np.all((0.05 <= np.asarray(action)) & (np.asarray(action) <= 0.95))
                assert model.outputs(state) == pytest.approx(reference, abs=1e-12)
                assert model.dynamics(
                    state,
                    action,
                    model.default_disturbances(),
                ) == pytest.approx([0.0, 0.0], abs=1e-12)
    finally:
        env.close()
    assert len({repr(spec) for spec in specs}) == len(specs)


def test_cstr_training_variation_is_seeded_and_records_scalar_disturbances():
    env = aiogym.make_env("cstr", randomize=True, disturbance=True)
    try:
        _, first = env.reset(seed=11)
        _, repeated = env.reset(seed=11)
        _, different = env.reset(seed=12)
        assert first["episode_spec"] == repeated["episode_spec"]
        assert first["episode_spec"] != different["episode_spec"]
        for values in first["episode_spec"]["disturbance_schedule"].values():
            assert all(np.isscalar(value) for value in values.values())
    finally:
        env.close()


def test_cstr_pid_uses_feed_for_concentration_and_cooling_for_temperature():
    env = aiogym.make_env("cstr")
    try:
        observation, info = env.reset(seed=0)
        model = env.unwrapped.model
        policy = aiogym.make_controller("pid", env=env)
        context = {"reference": [0.075, 72.0]}
        action = policy.act(observation, context)
        assert np.asarray(action).shape == (2,)
        assert float(action[0]) < model.default_action()[0]
        assert float(action[1]) < model.default_action()[1]
    finally:
        env.close()


@pytest.mark.parametrize(
    "benchmark",
    ("tracking", "disturbance-rejection", "boundary-safety"),
)
@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_cstr_baselines_complete_every_formal_horizon(benchmark, controller_id):
    env = aiogym.make_env("cstr", benchmark=benchmark)
    try:
        policy = aiogym.make_controller(controller_id, env=env)
        result = aiogym.evaluate(env=env, policy=policy, seeds=(0, 1, 2))
        aggregate = result["aggregate"]
        assert aggregate["safe_completion"]["mean"] == pytest.approx(1.0)
        assert aggregate["unsafe_rate"]["max"] == pytest.approx(0.0)
        assert aggregate["episode_length"]["min"] == pytest.approx(
            env.unwrapped.episode_steps
        )
        assert np.isfinite(aggregate["return"]["median"])
        assert np.isfinite(aggregate["tracking_iae"]["median"])
    finally:
        env.close()

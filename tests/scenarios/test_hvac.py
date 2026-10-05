from __future__ import annotations

import numpy as np
import pytest

import aiogym


def test_hvac_default_environment_starts_at_a_model_consistent_equilibrium():
    env = aiogym.make_env("hvac")
    try:
        observation, info = env.reset(seed=3)
        model = env.unwrapped.model
        state = model.initial_state()
        action = model.default_action()
        derivative = model.dynamics(state, action, model.default_disturbances())

        assert env.observation_space.shape == (4,)
        assert env.action_space.shape == (2,)
        assert env.observation_space.contains(observation)
        assert state == pytest.approx([22.0, 22.0])
        assert action == pytest.approx([0.7125, 0.7125])
        assert derivative == pytest.approx([0.0, 0.0], abs=1e-12)
        assert info["y"] == pytest.approx([22.0, 22.0])
    finally:
        env.close()


def test_hvac_parameter_overrides_are_strict_and_change_the_dynamics():
    base = aiogym.make_env("hvac")
    changed = aiogym.make_env(
        "hvac",
        parameters={"maximum_zone_power": 2400.0},
    )
    try:
        state = [22.0, 22.0]
        action = [1.0, 1.0]
        disturbances = base.unwrapped.model.default_disturbances()
        assert not np.allclose(
            base.unwrapped.model.dynamics(state, action, disturbances),
            changed.unwrapped.model.dynamics(state, action, disturbances),
        )
        assert changed.unwrapped.model.resolved_parameters[
            "maximum_zone_power"
        ] == pytest.approx(2400.0)
    finally:
        base.close()
        changed.close()

    with pytest.raises(ValueError, match="unknown hvac parameters"):
        aiogym.make_env("hvac", parameters={"unknown": 1.0})
    with pytest.raises(ValueError, match="finite"):
        aiogym.make_env(
            "hvac",
            parameters={"zone_thermal_capacity": float("nan")},
        )
    with pytest.raises(ValueError, match="positive"):
        aiogym.make_env(
            "hvac",
            parameters={"maximum_zone_power": 0.0},
        )


def test_hvac_tracking_cases_start_tracking_one_feasible_target():
    env = aiogym.make_env("hvac", benchmark="tracking")
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
            assert episode.horizon == 60
            assert not episode.reference_schedule
            assert np.all((19.0 <= values) & (values <= 25.0))
            assert np.all(np.abs(values[1] - values[0]) >= 1.5)
            assert episode.initial_state == pytest.approx(references[0])
            assert episode.initial_action == pytest.approx(
                model.tracking_steady_state_action(references[0])
            )
            for reference in references:
                action = model.tracking_steady_state_action(reference)
                state = model.tracking_steady_state_state(reference)
                assert action is not None
                assert state is not None
                assert np.all((0.05 <= np.asarray(action)) & (np.asarray(action) <= 0.95))
                assert model.dynamics(
                    state,
                    action,
                    model.default_disturbances(),
                ) == pytest.approx([0.0, 0.0], abs=1e-12)
    finally:
        env.close()
    assert len({repr(spec) for spec in specs}) == len(specs)


def test_hvac_training_variation_is_seeded_and_records_scalar_disturbances():
    env = aiogym.make_env("hvac", randomize=True, disturbance=True)
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


@pytest.mark.parametrize(
    "benchmark",
    ("tracking", "disturbance-rejection", "boundary-safety"),
)
@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_hvac_baselines_complete_every_formal_horizon(benchmark, controller_id):
    env = aiogym.make_env("hvac", benchmark=benchmark)
    try:
        policy = aiogym.make_controller(controller_id, env=env)
        result = aiogym.evaluate(env=env, policies={"policy": policy}, seeds=(0, 1, 2))["evaluations"]["policy"]
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

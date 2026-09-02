from __future__ import annotations

import numpy as np
import pytest

import aiogym


def test_heater_default_environment_starts_at_an_exact_equilibrium():
    env = aiogym.make_env("heater")
    try:
        observation, info = env.reset(seed=3)
        model = env.unwrapped.model
        state = model.initial_state()
        action = model.default_action()
        derivative = model.dynamics(state, action, model.default_disturbances())

        assert env.observation_space.shape == (5,)
        assert env.action_space.shape == (2,)
        assert env.observation_space.contains(observation)
        assert state == pytest.approx([758.7142857142858, 370.0, 3.0])
        assert action == pytest.approx(
            [0.3838817854996101, 0.7582745487136417]
        )
        assert model.outputs(state) == pytest.approx([3.0, 370.0])
        assert derivative == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
        assert info["y"] == pytest.approx([3.0, 370.0])
    finally:
        env.close()


def test_heater_uses_air_then_fuel_actions():
    env = aiogym.make_env("heater")
    try:
        model = env.unwrapped.model
        assert [row["name"] for row in model.action_schema()] == [
            "air_damper",
            "fuel_valve",
        ]
        combustion = model.combustion([0.5, 0.25])
        assert combustion["air_flow"] == pytest.approx(20.0)
        assert combustion["fuel_flow"] == pytest.approx(0.25)
        assert combustion["combustion_completeness"] == pytest.approx(1.0)
        env.reset(seed=0)
        with pytest.raises(ValueError, match="action"):
            env.step(np.asarray([0.5], dtype=np.float32))
        _, _, terminated, truncated, info = env.step(
            np.asarray([0.4, 0.7], dtype=np.float32)
        )
        assert not terminated
        assert not truncated
        assert info["applied_action"] == pytest.approx([0.4, 0.7])
    finally:
        env.close()


def test_heater_parameter_overrides_are_strict_and_change_the_dynamics():
    base = aiogym.make_env("heater")
    changed = aiogym.make_env(
        "heater",
        parameters={"heat_transfer_coefficient": 40_000.0},
    )
    try:
        state = base.unwrapped.model.initial_state()
        action = [0.4, 0.7]
        assert not np.allclose(
            base.unwrapped.model.dynamics(
                state,
                action,
                base.unwrapped.model.default_disturbances(),
            ),
            changed.unwrapped.model.dynamics(
                state,
                action,
                changed.unwrapped.model.default_disturbances(),
            ),
        )
        assert changed.unwrapped.model.resolved_parameters[
            "heat_transfer_coefficient"
        ] == pytest.approx(40_000.0)
    finally:
        base.close()
        changed.close()

    with pytest.raises(ValueError, match="unknown heater parameters"):
        aiogym.make_env("heater", parameters={"UA": 40_000.0})
    with pytest.raises(ValueError, match="finite"):
        aiogym.make_env(
            "heater",
            parameters={"maximum_air_flow": float("nan")},
        )
    with pytest.raises(ValueError, match="positive"):
        aiogym.make_env("heater", parameters={"firebox_heat_capacity": 0.0})
    with pytest.raises(ValueError, match="within"):
        aiogym.make_env("heater", parameters={"minimum_safe_oxygen": 21.0})


def test_heater_distinguishes_state_observation_and_controlled_outputs():
    env = aiogym.make_env("heater")
    try:
        model = env.unwrapped.model
        assert [row["name"] for row in model.state_schema()] == [
            "firebox_temperature",
            "outlet_temperature",
            "flue_oxygen",
        ]
        assert [row["name"] for row in model.output_schema()] == [
            "flue_oxygen",
            "outlet_temperature",
        ]
        assert len(model.observation_schema()) == 5
    finally:
        env.close()


def test_heater_tracking_cases_start_tracking_one_feasible_target():
    env = aiogym.make_env("heater", benchmark="tracking")
    specs = []
    try:
        model = env.unwrapped.model
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            start_reference = model.outputs(episode.initial_state)
            references = (start_reference, episode.reference)
            assert episode.horizon == 300
            assert not episode.reference_schedule
            difference = np.abs(
                np.asarray(episode.reference) - np.asarray(start_reference)
            )
            assert difference[0] >= 0.4
            assert difference[1] >= 2.0
            for reference in references:
                action = model.tracking_steady_state_action(reference)
                state = model.tracking_steady_state_state(reference)
                assert action is not None
                assert state is not None
                assert np.all(
                    (0.05 <= np.asarray(action))
                    & (np.asarray(action) <= 0.95)
                )
                assert model.outputs(state) == pytest.approx(
                    reference, abs=1e-10
                )
                assert model.dynamics(
                    state,
                    action,
                    model.default_disturbances(),
                ) == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)
    finally:
        env.close()
    assert len({repr(spec) for spec in specs}) == len(specs)


def test_heater_formal_disturbance_keeps_the_target_feasible():
    env = aiogym.make_env("heater", benchmark="disturbance-rejection")
    try:
        for seed in range(20):
            env.reset(seed=seed)
            model = env.unwrapped.model
            episode = env.unwrapped.episode
            disturbed = episode.disturbance_schedule[
                min(episode.disturbance_schedule)
            ]
            action = model.tracking_steady_state_action(
                episode.reference,
                disturbed,
            )
            state = model.tracking_steady_state_state(
                episode.reference,
                disturbed,
            )
            assert action is not None
            assert state is not None
            assert np.all(
                (0.05 <= np.asarray(action)) & (np.asarray(action) <= 0.95)
            )
            assert model.dynamics(state, action, disturbed) == pytest.approx(
                [0.0, 0.0, 0.0], abs=1e-10
            )
    finally:
        env.close()


def test_heater_training_variation_is_seeded_and_has_scalar_disturbances():
    env = aiogym.make_env("heater", randomize=True, disturbance=True)
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


def test_heater_pid_and_mpc_return_two_physical_actions():
    env = aiogym.make_env("heater")
    try:
        observation, info = env.reset(seed=0)
        for controller_id in ("pid", "mpc"):
            policy = aiogym.make_controller(controller_id, env=env)
            action = policy.act(observation, {"reference": info["reference"]})
            assert np.asarray(action).shape == (2,)
            assert env.action_space.contains(np.asarray(action, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize(
    "benchmark",
    ("tracking", "disturbance-rejection", "boundary-safety"),
)
@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_heater_baselines_complete_every_formal_horizon(
    benchmark,
    controller_id,
):
    env = aiogym.make_env("heater", benchmark=benchmark)
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

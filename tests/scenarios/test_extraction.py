from __future__ import annotations

import numpy as np
import pytest

import aiogym


def test_extraction_default_environment_starts_at_an_exact_equilibrium():
    env = aiogym.make_env("extraction")
    try:
        observation, info = env.reset(seed=3)
        model = env.unwrapped.model
        state = model.initial_state()
        action = model.default_action()
        derivative = model.dynamics(state, action, model.default_disturbances())

        assert env.observation_space.shape == (11,)
        assert env.action_space.shape == (2,)
        assert env.observation_space.contains(observation)
        assert model.outputs(state) == pytest.approx([0.30], abs=1e-12)
        assert action == pytest.approx([0.32051920093713704, 0.5])
        assert model.physical_flows(action) == pytest.approx(
            [163.65700446388283, 505.0]
        )
        assert derivative == pytest.approx([0.0] * 10, abs=1e-10)
        assert info["y"] == pytest.approx([0.30])
    finally:
        env.close()


def test_extraction_uses_two_normalized_flow_actions():
    env = aiogym.make_env("extraction")
    try:
        model = env.unwrapped.model
        assert model.physical_flows([0.0, 0.0]) == pytest.approx([5.0, 10.0])
        assert model.physical_flows([1.0, 1.0]) == pytest.approx(
            [500.0, 1000.0]
        )
        env.reset(seed=0)
        with pytest.raises(ValueError, match="action"):
            env.step(np.asarray([0.5], dtype=np.float32))
        _, _, terminated, truncated, info = env.step(
            np.asarray([0.45, 0.55], dtype=np.float32)
        )
        assert not terminated
        assert not truncated
        assert info["applied_action"] == pytest.approx([0.45, 0.55])
    finally:
        env.close()


def test_extraction_parameter_overrides_are_strict_and_change_the_dynamics():
    base = aiogym.make_env("extraction")
    changed = aiogym.make_env(
        "extraction",
        parameters={"mass_transfer_coefficient": 4.0},
    )
    try:
        state = base.unwrapped.model.initial_state()
        action = [0.4, 0.5]
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
            "mass_transfer_coefficient"
        ] == pytest.approx(4.0)
    finally:
        base.close()
        changed.close()

    with pytest.raises(ValueError, match="unknown extraction parameters"):
        aiogym.make_env("extraction", parameters={"Kla": 4.0})
    with pytest.raises(ValueError, match="finite"):
        aiogym.make_env(
            "extraction",
            parameters={"liquid_stage_volume": float("nan")},
        )
    with pytest.raises(ValueError, match="positive"):
        aiogym.make_env(
            "extraction",
            parameters={"gas_stage_volume": 0.0},
        )
    with pytest.raises(ValueError, match="requires"):
        aiogym.make_env(
            "extraction",
            parameters={"maximum_liquid_flow": 5.0},
        )


def test_extraction_controls_only_the_stage_5_liquid_quality():
    env = aiogym.make_env("extraction")
    try:
        model = env.unwrapped.model
        assert [row["name"] for row in model.output_schema()] == [
            "stage_5_liquid_concentration"
        ]
        assert len(model.state_schema()) == 10
        assert len(model.observation_schema()) == 11
    finally:
        env.close()

def test_extraction_tracking_cases_start_tracking_one_feasible_target():
    env = aiogym.make_env("extraction", benchmark="tracking")
    specs = []
    try:
        model = env.unwrapped.model
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            start_reference = model.outputs(episode.initial_state)
            references = (start_reference, episode.reference)
            values = np.asarray(references, dtype=float).reshape(-1)
            assert episode.horizon == 100
            assert not episode.reference_schedule
            assert values[0] == pytest.approx(0.30)
            assert np.all((0.12 <= values[1:]) & (values[1:] <= 0.44))
            assert abs(values[1] - values[0]) >= 0.05
            assert episode.initial_state == pytest.approx(model.initial_state())
            assert episode.initial_action == pytest.approx(model.default_action())
            for reference in references:
                action = model.tracking_steady_state_action(reference)
                state = model.tracking_steady_state_state(reference)
                assert action is not None
                assert state is not None
                assert action[1] == pytest.approx(0.5)
                assert np.all((0.05 <= np.asarray(action)) & (np.asarray(action) <= 0.95))
                assert model.outputs(state) == pytest.approx(reference, abs=1e-10)
                assert model.dynamics(
                    state,
                    action,
                    model.default_disturbances(),
                ) == pytest.approx([0.0] * 10, abs=1e-10)
    finally:
        env.close()
    assert len({repr(spec) for spec in specs}) == len(specs)


def test_extraction_training_variation_is_seeded_and_has_scalar_disturbances():
    env = aiogym.make_env("extraction", randomize=True, disturbance=True)
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


def test_extraction_pid_holds_the_gas_feed_at_its_nominal_bias():
    env = aiogym.make_env("extraction")
    try:
        observation, info = env.reset(seed=0)
        policy = aiogym.make_controller("pid", env=env)
        action = policy.act(observation, {"info": info})
        assert np.asarray(action).shape == (2,)
        assert float(action[1]) == pytest.approx(0.5)
    finally:
        env.close()


@pytest.mark.parametrize(
    "benchmark",
    ("tracking", "disturbance-rejection", "boundary-safety"),
)
@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_extraction_baselines_complete_every_formal_horizon(
    benchmark,
    controller_id,
):
    env = aiogym.make_env("extraction", benchmark=benchmark)
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

from __future__ import annotations

import numpy as np
import pytest

import aiogym
from aiogym.scenarios.crystallization.model import CrystallizationModel


def test_crystallization_default_batch_matches_the_model_endpoint():
    env = aiogym.make_env("crystallization")
    try:
        observation, info = env.reset(seed=3)
        model = env.unwrapped.model
        endpoint = model.batch_endpoint(
            env.unwrapped.episode.initial_action,
            horizon_steps=env.unwrapped.episode_steps,
        )

        assert env.observation_space.shape == (7,)
        assert env.action_space.shape == (1,)
        assert env.observation_space.contains(observation)
        assert model.outputs(model.initial_state()) == pytest.approx(
            [0.333333333325926, 14.999999984999999]
        )
        assert env.unwrapped.episode.reference == pytest.approx(
            endpoint["output"], abs=1e-12
        )
        assert info["y"] == pytest.approx(model.outputs(model.initial_state()))
    finally:
        env.close()


def test_crystallization_environment_integration_matches_batch_endpoint():
    env = aiogym.make_env("crystallization")
    try:
        observation, _ = env.reset(seed=0)
        del observation
        model = env.unwrapped.model
        action = np.asarray(env.unwrapped.episode.initial_action, dtype=np.float32)
        endpoint = model.batch_endpoint(
            action,
            horizon_steps=env.unwrapped.episode_steps,
        )
        for step in range(env.unwrapped.episode_steps):
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated
            assert truncated == (step == env.unwrapped.episode_steps - 1)
        assert env.unwrapped.state == pytest.approx(endpoint["state"], abs=1e-10)
        assert info["y"] == pytest.approx(endpoint["output"], abs=1e-10)
    finally:
        env.close()


def test_crystallization_action_maps_to_cooling_temperature():
    model = CrystallizationModel()
    assert model.cooling_temperature([0.0]) == pytest.approx(30.0)
    assert model.cooling_temperature([0.5]) == pytest.approx(35.0)
    assert model.cooling_temperature([1.0]) == pytest.approx(40.0)


def test_crystallization_parameter_overrides_are_strict_and_change_dynamics():
    base = aiogym.make_env("crystallization")
    changed = aiogym.make_env(
        "crystallization",
        parameters={"growth_rate_coefficient": 40.0},
    )
    try:
        state = base.unwrapped.model.initial_state()
        action = [0.5]
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
            "growth_rate_coefficient"
        ] == pytest.approx(40.0)
    finally:
        base.close()
        changed.close()

    with pytest.raises(ValueError, match="unknown crystallization parameters"):
        aiogym.make_env("crystallization", parameters={"kg": 40.0})
    with pytest.raises(ValueError, match="finite"):
        aiogym.make_env(
            "crystallization",
            parameters={"growth_rate_scale": float("nan")},
        )
    with pytest.raises(ValueError, match="positive"):
        aiogym.make_env(
            "crystallization",
            parameters={"maximum_growth_rate": 0.0},
        )
    with pytest.raises(ValueError, match="minimum < maximum"):
        aiogym.make_env(
            "crystallization",
            parameters={"maximum_cooling_temperature": 30.0},
        )


def test_crystallization_is_a_batch_process_without_tracking_equilibrium():
    env = aiogym.make_env("crystallization")
    try:
        model = env.unwrapped.model
        reference = model.default_setpoint_vector()
        assert model.tracking_steady_state_action(reference) is None
        assert model.tracking_steady_state_state(reference) is None
        for action in ([0.0], [0.5], [1.0]):
            derivative = np.asarray(
                model.dynamics(
                    model.initial_state(),
                    action,
                    model.default_disturbances(),
                )
            )
            assert np.any(np.abs(derivative) > 0.0)
    finally:
        env.close()


def test_crystallization_distinguishes_state_observation_and_quality_output():
    env = aiogym.make_env("crystallization")
    try:
        model = env.unwrapped.model
        assert len(model.state_schema()) == 5
        assert [row["name"] for row in model.output_schema()] == [
            "coefficient_of_variation",
            "mean_crystal_size",
        ]
        assert len(model.observation_schema()) == 7
    finally:
        env.close()


def test_crystallization_tracking_seeds_resolve_distinct_reachable_targets():
    env = aiogym.make_env("crystallization", benchmark="tracking")
    specs = []
    try:
        model = env.unwrapped.model
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            specs.append(info["episode_spec"])
            assert episode.horizon == 50
            assert not episode.reference_schedule
            assert 0.80 <= episode.initial_action[0] <= 0.90
            endpoint = model.batch_endpoint(
                episode.initial_action,
                horizon_steps=episode.horizon,
                initial_state=episode.initial_state,
                disturbances=episode.disturbances,
            )
            assert endpoint["output"] == pytest.approx(
                episode.reference, abs=1e-10
            )
    finally:
        env.close()
    assert len({repr(spec) for spec in specs}) == len(specs)


def test_crystallization_boundary_case_starts_near_concentration_limit():
    env = aiogym.make_env("crystallization", benchmark="boundary-safety")
    try:
        _, info = env.reset(seed=0)
        concentration = env.unwrapped.episode.initial_state[-1]
        assert 1.70 <= concentration <= 1.95
        assert info["constraint_costs"] == pytest.approx(
            {name: 0.0 for name in info["constraint_costs"]}
        )
        assert info["safety_margins"]["solute_concentration_upper"] == (
            pytest.approx((2.0 - concentration) / 2.0)
        )
    finally:
        env.close()


def test_crystallization_training_variation_is_seeded_and_scalar():
    env = aiogym.make_env(
        "crystallization",
        randomize=True,
        disturbance=True,
    )
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


def test_crystallization_pid_and_mpc_return_one_physical_action():
    env = aiogym.make_env("crystallization")
    try:
        observation, info = env.reset(seed=0)
        for controller_id in ("pid", "mpc"):
            policy = aiogym.make_controller(controller_id, env=env)
            action = policy.act(observation, {"reference": info["reference"]})
            assert np.asarray(action).shape == (1,)
            assert env.action_space.contains(np.asarray(action, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize(
    "benchmark",
    ("tracking", "disturbance-rejection", "boundary-safety"),
)
@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_crystallization_baselines_complete_every_formal_batch(
    benchmark,
    controller_id,
):
    env = aiogym.make_env("crystallization", benchmark=benchmark)
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
        assert np.isfinite(aggregate["final_error"]["median"])
    finally:
        env.close()

from __future__ import annotations

import math

import numpy as np
import pytest

import aiogym
from aiogym._internal.identifiers import internal_scenario_id
from aiogym.models.metadata import iter_model_metadata
from aiogym.models.registry import make_model
from aiogym.models.validation import validate_model_readiness


EXPECTED_SCENARIOS = {
    "cascade",
    "cascade-recirculating",
    "crystallization",
    "cstr",
    "extraction",
    "heater",
    "hvac",
    "quadruple",
}
SCENARIOS = tuple(sorted(EXPECTED_SCENARIOS))

# Keep any model-specific smoke concessions here instead of adding runtime
# branches. All current built-ins satisfy the shared deterministic contract.
SCENARIO_ENVIRONMENT_OVERRIDES: dict[str, dict[str, object]] = {}


def _environment_config(scenario: str) -> dict[str, object]:
    environment: dict[str, object] = {
        "episode_steps": 2,
        "auto_events": False,
        "randomize": False,
        "randomize_setpoints": False,
        "randomize_plant": False,
        "plant_drift": False,
        "noise": False,
    }
    environment.update(SCENARIO_ENVIRONMENT_OVERRIDES.get(scenario, {}))
    return {"scenario": scenario, "environment": environment}


def _neutral_action(space) -> np.ndarray:
    """Return a deterministic finite point inside a Box action space."""

    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    action = np.zeros(space.shape, dtype=np.float64)
    finite_low = np.isfinite(low)
    finite_high = np.isfinite(high)
    bounded = finite_low & finite_high
    lower_bounded = finite_low & ~finite_high
    upper_bounded = ~finite_low & finite_high

    action[bounded] = 0.5 * low[bounded] + 0.5 * high[bounded]
    action[lower_bounded] = np.maximum(low[lower_bounded], 0.0)
    action[upper_bounded] = np.minimum(high[upper_bounded], 0.0)
    action = action.astype(space.dtype, copy=False)
    assert space.contains(action)
    assert np.all(np.isfinite(action))
    return action


def _two_step_rollout(scenario: str):
    env = aiogym.make_env(config=_environment_config(scenario))
    try:
        observation, reset_info = env.reset(seed=123)
        action = _neutral_action(env.action_space)
        assert isinstance(reset_info, dict)
        assert env.observation_space.contains(observation)
        assert np.all(np.isfinite(observation))
        assert np.all(np.isfinite(env.unwrapped.integ.x))

        transitions = []
        for _ in range(2):
            result = env.step(action.copy())
            assert isinstance(result, tuple)
            assert len(result) == 5
            next_observation, reward, terminated, truncated, info = result
            assert env.observation_space.contains(next_observation)
            assert np.all(np.isfinite(next_observation))
            assert math.isfinite(float(reward))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)
            true_state = np.asarray(
                env.unwrapped.integ.x,
                dtype=np.float64,
            )
            assert np.all(np.isfinite(true_state))
            transitions.append(
                (
                    np.asarray(next_observation).copy(),
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    true_state.copy(),
                )
            )
        return np.asarray(observation).copy(), action.copy(), transitions
    finally:
        env.close()


def test_canonical_scenario_registry_is_complete():
    assert set(aiogym.list_scenarios()) == EXPECTED_SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenario_models_satisfy_the_metadata_contract(scenario):
    model = make_model(scenario)
    initial_state = np.asarray(model.initial_state(), dtype=np.float64)

    assert model.scenario == internal_scenario_id(scenario)
    assert len(model.state_schema()) == len(initial_state)
    assert len(model.action_schema()) == model.action_dim()
    assert np.all(np.isfinite(initial_state))

    metadata_scenario, metadata = next(iter_model_metadata((scenario,)))
    assert metadata_scenario == scenario
    assert metadata["scenario"] == scenario
    assert metadata["state_vector"]["length"] == len(initial_state)
    assert metadata["action_vector"]["length"] == model.action_dim()

    readiness = validate_model_readiness(model)
    assert readiness["passed"], readiness


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenario_environments_have_deterministic_two_step_smoke(scenario):
    first_observation, first_action, first_steps = _two_step_rollout(
        scenario
    )
    second_observation, second_action, second_steps = _two_step_rollout(
        scenario
    )

    np.testing.assert_array_equal(first_action, second_action)
    np.testing.assert_allclose(
        first_observation,
        second_observation,
        rtol=0.0,
        atol=np.finfo(first_observation.dtype).eps,
    )
    for first, second in zip(first_steps, second_steps):
        first_obs, first_reward, first_terminated, first_truncated, first_state = (
            first
        )
        second_obs, second_reward, second_terminated, second_truncated, second_state = (
            second
        )
        np.testing.assert_allclose(
            first_obs,
            second_obs,
            rtol=0.0,
            atol=np.finfo(first_obs.dtype).eps,
        )
        assert first_reward == pytest.approx(
            second_reward,
            rel=0.0,
            abs=1e-12,
        )
        assert first_terminated is second_terminated
        assert first_truncated is second_truncated
        np.testing.assert_allclose(
            first_state,
            second_state,
            rtol=0.0,
            atol=1e-12,
        )

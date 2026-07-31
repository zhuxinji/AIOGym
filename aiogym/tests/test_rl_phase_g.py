"""Phase-G acceptance tests for realism, memory, and safety."""
from __future__ import annotations

import copy

import numpy as np
import pytest

import aiogym
from aiogym.controllers.adapters import PolicyController
from aiogym.evaluation.execution import evaluate_controller
from aiogym.evaluation.safety_gate import (
    SafetyGateSpec,
    evaluate_safety_gate,
)
from aiogym.evaluation.statistics import build_intervention_report
from aiogym.generation.samplers import episode_spec_from_case
from aiogym.generation.specs import EpisodeSpec
from aiogym.experimental.rl import (
    LagrangeMultiplier,
    LagrangianSAC,
    ObservationContract,
    ProjectionSafetyShield,
    RecurrentStateContract,
    SafetyShieldWrapper,
    wrap_observation_contract,
)
from aiogym.rl.statistics import NormalizedActionWrapper
from aiogym.tests._env import make_test_env


def test_sensor_and_actuator_models_are_reproducible_and_audited():
    case = aiogym.load_case("quadruple/minimum-phase")
    base = episode_spec_from_case(
        case,
        seed=31,
        scenario="quadruple",
        goal="regulation",
        split="validation",
    )
    payload = base.payload
    payload["sensor_model"] = {
        "kind": "sensor_dynamics_v1",
        "noise_fraction": 0.002,
        "bias_fraction": [0.01, -0.01, 0.0, 0.0],
        "drift_fraction_per_second": 0.0,
        "delay_steps": 1,
        "quantization_fraction": 0.001,
        "dropout_probability": 0.0,
    }
    payload["actuator_model"] = {
        "kind": "actuator_dynamics_v1",
        "efficiency": [0.95, 0.9],
        "bias": 0.0,
        "delay_steps": 1,
        "deadband": 0.01,
        "time_constant_seconds": 2.0,
        "slew_rate_per_second": 0.05,
    }
    episode = EpisodeSpec(payload)

    def rollout_once():
        env = make_test_env(
            "quadruple",
            case=case,
            reward_spec="regulation-v1",
        )
        try:
            observation, _ = env.reset(
                options={"episode_spec": episode}
            )
            next_observation, _, _, _, info = env.step(
                np.ones(env.action_space.shape, dtype=np.float32)
            )
            return (
                observation,
                next_observation,
                np.asarray(env.last_commanded_act),
                np.asarray(env.last_act),
                info,
            )
        finally:
            env.close()

    first = rollout_once()
    second = rollout_once()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_allclose(first[2], np.ones(2))
    assert not np.allclose(first[2], first[3])
    assert first[4]["actuator_intervened"]
    assert np.asarray(first[4]["action_commanded_physical"]).shape == (2,)
    assert np.asarray(first[4]["action_applied_physical"]).shape == (2,)


def test_measured_output_history_and_recurrent_contracts_reset_cleanly():
    env = make_test_env(
        "quadruple",
        case="minimum-phase",
        reward_spec="regulation-v1",
        observation_mode="measured_output",
        disturbance_obs=False,
        previous_action_obs=False,
        normalize_observations=False,
        tracking_error_obs=False,
        integral_obs=False,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        episode_steps=2,
    )
    contract = ObservationContract(
        sensing="measured_output",
        temporal="history",
        history_length=3,
        include_action_history=True,
    )
    wrapped = wrap_observation_contract(
        NormalizedActionWrapper(env),
        contract,
    )
    try:
        observation, _ = wrapped.reset(seed=4)
        assert env.observation_space.shape == (4,)
        assert observation.shape == wrapped.observation_space.shape == (16,)
        np.testing.assert_array_equal(
            observation[:4],
            observation[4:8],
        )
        np.testing.assert_array_equal(
            observation[4:8],
            observation[8:12],
        )
    finally:
        wrapped.close()

    recurrent = RecurrentStateContract((2, 8))
    state = np.ones((3, 2, 8), dtype=np.float32)
    reset = recurrent.reset_where(state, [False, True, False])
    assert np.all(reset[1] == 0.0)
    assert np.all(reset[[0, 2]] == 1.0)


def test_learned_policy_context_does_not_expose_environment():
    seen = {}

    class Policy:
        def act(self, observation, context):
            seen["env"] = context.env
            seen["info"] = context.info
            seen["measurement"] = context.measurement
            return np.full(2, 0.5, dtype=np.float32)

    env = make_test_env(
        "quadruple",
        case="minimum-phase",
        reward_spec="regulation-v1",
        observation_mode="measured_output",
        episode_steps=1,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
    )
    try:
        evaluate_controller(PolicyController(Policy()), env, seed=2)
    finally:
        env.close()
    assert seen["env"] is None
    assert seen["info"] == {}
    assert seen["measurement"]["observation_mode"] == "measured_output"
    assert set(seen["measurement"]) == {"y", "observation_mode"}


def test_safety_shield_keeps_raw_proposal_and_reports_intervention():
    base = make_test_env(
        "quadruple",
        case="minimum-phase",
        reward_spec="regulation-v1",
        episode_steps=2,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env = SafetyShieldWrapper(
        base,
        ProjectionSafetyShield(
            low=0.2,
            high=0.8,
            max_delta_per_step=0.1,
        ),
    )
    try:
        env.reset(seed=5)
        _, _, _, _, first = env.step(np.ones(2, dtype=np.float32))
        _, _, _, _, second = env.step(np.zeros(2, dtype=np.float32))
    finally:
        env.close()
    np.testing.assert_array_equal(
        first["action_policy_proposed"],
        np.ones(2, dtype=np.float32),
    )
    np.testing.assert_allclose(first["action_shielded"], [0.8, 0.8])
    np.testing.assert_allclose(second["action_shielded"], [0.7, 0.7])
    assert first["shield_intervened"]
    assert "action_bounds" in first["shield_intervention_reasons"]
    assert "action_slew" in second["shield_intervention_reasons"]
    assert second["costs"]["protection_intervention"] > 0.0


def test_evaluator_accumulates_shield_interventions_without_hiding_policy():
    class Policy:
        def predict(self, observation, deterministic=True):
            return np.ones(2, dtype=np.float32)

    base = make_test_env(
        "quadruple",
        case="minimum-phase",
        reward_spec="regulation-v1",
        episode_steps=2,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env = SafetyShieldWrapper(
        base,
        ProjectionSafetyShield(high=0.8),
    )
    try:
        result = evaluate_controller(
            Policy(),
            env,
            seed=9,
            include_episodes=True,
        )
    finally:
        env.close()
    assert result["shield_intervention_count"] == 2.0
    assert result["protection_intervention_count"] == 2.0
    assert result["episode_metrics"][0][
        "shield_intervention_magnitude"
    ] == pytest.approx(0.8)
    assert result["environment"]["safety_shield"]["kind"] == "projection"


def test_constrained_gate_applies_declared_cost_budgets():
    spec = SafetyGateSpec(
        cost_budgets={
            "soft_safety_cost": 1.0,
            "shield_intervention_count": 2.0,
        }
    )
    passed = evaluate_safety_gate(
        {
            "soft_safety_cost": 0.5,
            "shield_intervention_count": 2.0,
        },
        spec,
    )
    failed = evaluate_safety_gate(
        {
            "soft_safety_cost": 1.5,
            "shield_intervention_count": 2.0,
        },
        spec,
    )
    assert passed["eligible"]
    assert not failed["eligible"]
    assert failed["reasons"] == ["cost_budget:soft_safety_cost"]


def test_lagrangian_sac_separates_reward_and_cost_and_resumes():
    agent = LagrangianSAC(
        3,
        2,
        cost_limit=0.2,
        hidden=8,
        batch_size=4,
        replay_capacity=16,
        seed=7,
    )
    for index in range(6):
        agent.push(
            np.full(3, index, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            reward=1.0,
            cost=0.5,
            next_observation=np.full(3, index + 1, dtype=np.float32),
            terminated=False,
        )
    metrics = agent.update()
    assert {
        "reward_critic_loss",
        "cost_critic_loss",
        "lagrange_multiplier",
    } <= set(metrics)
    assert agent.multiplier.value > 0.0
    state = copy.deepcopy(agent.state_dict())
    restored = LagrangianSAC(
        3,
        2,
        cost_limit=0.2,
        hidden=8,
        batch_size=4,
        replay_capacity=16,
        seed=7,
    )
    restored.load_state_dict(state)
    assert restored.multiplier.state_dict() == agent.multiplier.state_dict()
    assert len(restored.replay) == len(agent.replay)
    assert restored.artifact_metadata()["reward_cost_separation"]


def test_intervention_report_contains_per_seed_per_case_safety_tails():
    evaluation = {
        "track_id": "track",
        "track_hash": "hash",
        "split": "test",
        "base_seeds": [10, 11],
        "results": [
            {
                "case_id": "case-a",
                "case_horizon_seconds": 10.0,
                "episode_metrics": [
                    {
                        "seed": 10,
                        "shield_intervention_count": 1.0,
                        "shield_intervention_duration": 2.0,
                        "shield_intervention_magnitude": 0.5,
                    },
                    {
                        "seed": 11,
                        "shield_intervention_count": 0.0,
                        "shield_intervention_duration": 0.0,
                        "shield_intervention_magnitude": 0.0,
                    },
                ],
            }
        ],
    }
    report = build_intervention_report({"agent": evaluation})
    shield = report["algorithms"]["agent"]["shield"]
    assert np.asarray(shield["rate_matrix"]).shape == (2, 1)
    assert shield["rate_matrix"][0][0] == pytest.approx(0.2)
    assert shield["summary"]["episode_intervention_probability"] == 0.5
    assert "cvar95_severity" in shield["summary"]


def test_lagrange_multiplier_projects_at_zero():
    multiplier = LagrangeMultiplier(
        cost_limit=1.0,
        learning_rate=0.5,
        initial_value=0.1,
    )
    assert multiplier.update(0.0) == 0.0

"""Contracts for pure stage-reward evaluation."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from aiogym.env import AIOGymNativeEnv
from aiogym import list_scenarios
from aiogym.models import apply_model_params, make_model

SCENARIO_IDS = list_scenarios()
from aiogym.evaluation.objectives import stage_reward

class StageRewardContractTests(unittest.TestCase):
    def test_continuous_cascade_penalizes_product_flow_shortfall(self):
        model = make_model("cascade")
        model.configure_operation({
            "product_flow_sp": 4.0e-4,
            "min_product_flow": 4.0e-4,
        })
        target = model.default_setpoint_vector()
        state = [target[0], target[3], target[1], target[4], target[2], target[5]]
        disturbance = model.runtime_env(model.disturbance_defaults())
        steady = model.steady_state_requirements(target, disturbance)
        context = {
            "setpoint": target,
            "disturbance": disturbance,
            "reward_mode": "economic",
            "reward_scale": 1.0,
            "tracking_q_y": [1.0] * 6,
            "tracking_r_move": 0.0,
            "terminate_on_runaway": False,
            "dt": 1.0,
        }
        stopped = stage_reward(
            model,
            state,
            [0.0] * 7,
            state,
            previous_action=[0.0] * 7,
            **context,
        )
        producing = stage_reward(
            model,
            state,
            steady["action"],
            state,
            previous_action=steady["action"],
            **context,
        )
        self.assertLess(stopped.info["profit"], producing.info["profit"])
        self.assertAlmostEqual(stopped.info["production"], 0.0)
        self.assertAlmostEqual(stopped.info["product_flow_shortfall_m3s"], 4.0e-4)
        self.assertAlmostEqual(producing.info["production"], 4.0e-4)
        self.assertAlmostEqual(producing.info["product_flow_shortfall_m3s"], 0.0)

        zero_target = make_model("cascade")
        zero_target_result = stage_reward(
            zero_target,
            state,
            [0.0] * 7,
            state,
            previous_action=[0.0] * 7,
            **context,
        )
        self.assertAlmostEqual(zero_target_result.info["profit"], 0.0)
        self.assertAlmostEqual(
            zero_target_result.info["product_flow_shortfall_m3s"], 0.0
        )

    def test_economic_terms_integrate_time_and_use_reported_action_energy(self):
        cstr = make_model("cstr")
        cstr_state = [0.5, 60.0]
        cstr_action = [1.0, 0.0]
        cstr_context = {
            "setpoint": cstr.default_setpoint_vector(),
            "disturbance": cstr.runtime_env(cstr.disturbance_defaults()),
            "previous_action": cstr_action,
            "reward_mode": "economic",
            "reward_scale": 1.0,
            "tracking_q_y": [1.0],
            "tracking_r_move": 0.0,
            "terminate_on_runaway": False,
        }
        half_second = stage_reward(
            cstr, cstr_state, cstr_action, cstr_state, dt=0.5, **cstr_context
        )
        one_second = stage_reward(
            cstr, cstr_state, cstr_action, cstr_state, dt=1.0, **cstr_context
        )
        self.assertAlmostEqual(one_second.info["production"], 2.0 * half_second.info["production"])
        self.assertAlmostEqual(one_second.info["profit"], 2.0 * half_second.info["profit"])
        self.assertAlmostEqual(one_second.reward, one_second.info["profit"])
        # Feed-pump power is part of action_energy_kw and must now also affect profit.
        expected_rate = 1575.0 * 0.01 - 0.7 * 1.2
        self.assertAlmostEqual(one_second.info["energy_kw"], 1.2)
        self.assertAlmostEqual(one_second.info["profit"], expected_rate)

        cascade = make_model("cascade")
        cascade_state = [0.45, 39.0, 0.45, 53.0, 0.45, 66.0]
        pump_only = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = stage_reward(
            cascade,
            cascade_state,
            pump_only,
            cascade_state,
            setpoint=cascade.default_setpoint_vector(),
            disturbance=cascade.runtime_env(cascade.disturbance_defaults()),
            previous_action=pump_only,
            reward_mode="economic",
            reward_scale=1.0,
            tracking_q_y=[1.0] * 6,
            tracking_r_move=0.0,
            terminate_on_runaway=False,
            dt=2.0,
        )
        self.assertAlmostEqual(result.info["energy_kw"], 1.5)
        self.assertAlmostEqual(result.info["profit"], -0.7 * 1.5 * 2.0)

    def test_stage_reward_matches_environment_step(self):
        for scenario in SCENARIO_IDS:
            for reward_mode in ("tracking", "kpi", "economic"):
                with self.subTest(scenario=scenario, reward_mode=reward_mode):
                    env = AIOGymNativeEnv(
                        scenario,
                        reward_mode=reward_mode,
                        auto_events=False,
                        randomize=False,
                        randomize_setpoints=False,
                        randomize_plant=False,
                        plant_drift=False,
                        episode_steps=2,
                    )
                    env.reset(seed=0)
                    state = list(env.integ.x)
                    previous_action = copy.deepcopy(env.previous_act)
                    setpoint = list(env.y_sp)
                    disturbance = copy.deepcopy(env._env())
                    action = np.full(env.action_space.shape, 0.4, dtype=np.float32)
                    physical_action = env._split(action)

                    _, environment_reward, terminated, _, environment_info = env.step(action)
                    score_after_step = env.scorer.report()
                    expected = stage_reward(
                        env.model,
                        state,
                        physical_action,
                        env.integ.x,
                        setpoint=setpoint,
                        disturbance=disturbance,
                        previous_action=previous_action,
                        reward_mode=reward_mode,
                        reward_scale=env.reward_scale,
                        tracking_q_y=env.tracking_q_y,
                        tracking_r_move=env.tracking_r_move,
                        terminate_on_runaway=env.terminate_on_runaway,
                        dt=env.control_dt,
                        economic_config=env._econ,
                    )

                    self.assertTrue(math.isclose(expected.reward, environment_reward, rel_tol=1e-12, abs_tol=1e-12))
                    self.assertIs(expected.terminated, terminated)
                    self.assertTrue(math.isclose(
                        expected.info["tracking_cost"],
                        environment_info["tracking_cost"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ))
                    self.assertTrue(math.isclose(
                        expected.info["energy_kw"],
                        environment_info["energy_kw"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ))
                    self.assertEqual(env.scorer.report(), score_after_step)

    def test_quadruple_tracking_cost_uses_identity_weights_after_normalization(self):
        model = apply_model_params(make_model("quadruple"), {
            "gamma": [0.43, 0.34],
            "nominal_voltage": [3.15, 3.15],
        })
        state = model.initial_state()
        setpoint = [state[0] + 1.0, state[1]]
        action = [0.4, 0.2]
        previous_action = [0.3, 0.3]
        result = stage_reward(
            model,
            state,
            action,
            state,
            setpoint=setpoint,
            disturbance=model.runtime_env(model.disturbance_defaults()),
            previous_action=previous_action,
            reward_mode="tracking",
            reward_scale=1.0,
            tracking_q_y=[1.0, 1.0],
            tracking_r_move=1.0,
            terminate_on_runaway=False,
        )

        self.assertAlmostEqual(result.info["tracking_move_cost"], 0.1 ** 2 + 0.1 ** 2)
        self.assertAlmostEqual(result.info["tracking_error_cost"], (1.0 / 20.0) ** 2)
        self.assertNotIn("tracking_steady_cost", result.info)
        self.assertNotIn("tracking_steady_action", result.info)
        self.assertAlmostEqual(
            result.info["tracking_cost"],
            result.info["tracking_error_cost"] + result.info["tracking_move_cost"],
        )
        self.assertAlmostEqual(result.reward, -result.info["tracking_cost"])

    def test_stage_reward_repeated_calls_are_side_effect_free(self):
        env = AIOGymNativeEnv(
            "cstr",
            reward_mode="tracking",
            auto_events=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        action = env._split(np.array([0.25, 0.75], dtype=np.float32))
        kwargs = {
            "setpoint": list(env.y_sp),
            "disturbance": copy.deepcopy(env._env()),
            "previous_action": copy.deepcopy(env.previous_act),
            "reward_mode": env.reward_mode,
            "reward_scale": env.reward_scale,
            "tracking_q_y": tuple(env.tracking_q_y),
            "tracking_r_move": env.tracking_r_move,
            "terminate_on_runaway": env.terminate_on_runaway,
            "economic_config": copy.deepcopy(env._econ),
        }
        scorer_before = env.scorer.report()
        previous_action_before = copy.deepcopy(env.previous_act)
        state_before = list(env.integ.x)

        first = stage_reward(env.model, state, action, state, **kwargs)
        second = stage_reward(env.model, state, action, state, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(env.scorer.report(), scorer_before)
        self.assertEqual(env.previous_act, previous_action_before)
        self.assertEqual(env.integ.x, state_before)

    def test_environment_evaluates_candidate_actions_with_explicit_context(self):
        env = AIOGymNativeEnv(
            "cstr",
            reward_mode="tracking",
            auto_events=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        candidate = np.array([0.2, 0.8], dtype=np.float32)
        scorer_before = env.scorer.report()
        state_before = list(env.integ.x)

        with_default_previous = env.evaluate_transition(state, candidate, state)
        with_candidate_previous = env.evaluate_transition(
            state,
            candidate,
            state,
            setpoint=list(env.y_sp),
            disturbance=copy.deepcopy(env._env()),
            previous_action=candidate,
        )

        self.assertGreater(with_default_previous.info["tracking_move_cost"], 0.0)
        self.assertEqual(with_candidate_previous.info["tracking_move_cost"], 0.0)
        self.assertEqual(env.scorer.report(), scorer_before)
        self.assertEqual(env.integ.x, state_before)

    def test_environment_rejects_setpoint_actions_for_transition_evaluation(self):
        env = AIOGymNativeEnv(
            "cstr",
            action_mode="setpoint",
            auto_events=False,
            randomize=False,
            randomize_setpoints=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)

        with self.assertRaisesRegex(ValueError, "action_mode='actuator'"):
            env.evaluate_transition(state, env.action_space.sample(), state)

    def test_custom_stage_reward_is_shared_by_step_and_prediction(self):
        calls = []

        def custom_stage_reward(state, action, next_state, context):
            calls.append((state, action, next_state, context))
            context.disturbance["local_only"] = 1.0
            context.info["local_only"] = True
            return 4.0 - abs(next_state[0] - context.setpoint[0])

        env = AIOGymNativeEnv(
            "cstr",
            reward_mode="tracking",
            custom_stage_reward=custom_stage_reward,
            auto_events=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        previous_action = copy.deepcopy(env.previous_act)
        setpoint = list(env.y_sp)
        disturbance = copy.deepcopy(env._env())
        action = np.array([0.3, 0.7], dtype=np.float32)

        _, environment_reward, _, _, environment_info = env.step(action)
        predicted = env.evaluate_transition(
            state,
            action,
            env.integ.x,
            setpoint=setpoint,
            disturbance=disturbance,
            previous_action=previous_action,
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(math.isclose(environment_reward, predicted.reward, rel_tol=1e-12, abs_tol=1e-12))
        self.assertNotIn("local_only", environment_info)
        self.assertNotIn("local_only", env._env())

    def test_custom_stage_reward_receives_terminal_penalty_once(self):
        env = AIOGymNativeEnv(
            "cstr",
            custom_stage_reward=lambda *_: 7.0,
            terminate_on_runaway=True,
            auto_events=False,
            randomize=False,
            randomize_setpoints=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        result = env.evaluate_transition(
            state,
            np.array([0.5, 0.5], dtype=np.float32),
            [state[0], 100.0],
        )

        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, -43.0)

    def test_custom_stage_reward_rejects_invalid_result(self):
        env = AIOGymNativeEnv(
            "cstr",
            custom_stage_reward=lambda *_: float("nan"),
            auto_events=False,
            randomize=False,
            randomize_setpoints=False,
        )
        env.reset(seed=0)
        state = list(env.integ.x)
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            env.evaluate_transition(state, np.array([0.5, 0.5], dtype=np.float32), state)

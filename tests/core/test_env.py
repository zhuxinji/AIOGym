from __future__ import annotations

import numpy as np
import pytest

import aiogym
from aiogym.core.contracts import Scenario
from aiogym.core.env import ProcessControlEnv, make_env
from aiogym.core.registry import register_scenario, unregister_scenario
from aiogym.core.specs import Benchmark, EpisodeSpec, Reward


class ToyModel:
    scenario = "core-toy"
    dt_micro = 0.5

    def __init__(self, parameters=None):
        if parameters:
            raise ValueError("core toy has no parameters")
        self.resolved_parameters = {}

    def initial_state(self):
        return [0.0]

    def dynamics(self, state, action, disturbances=None):
        del disturbances
        return [float(action[0]) - float(state[0])]

    def outputs(self, state):
        return list(state)

    def action_schema(self):
        return [{"name": "u", "low": 0.0, "high": 1.0}]

    def state_schema(self):
        return [{"name": "x", "low": -10.0, "high": 10.0}]

    def default_action(self):
        return [0.0]

    def default_setpoint_vector(self):
        return [1.0]

    def default_disturbances(self):
        return {}

    def output_schema(self):
        return [{"name": "x", "low": -10.0, "high": 10.0}]

    def observation_schema(self):
        return [{**row, "kind": "measurement"} for row in self.output_schema()]

    def action_slew_limits(self):
        return None

    def clamp_state(self, state):
        return state

    def observation(self, state, reference, previous_action, disturbances):
        del reference, previous_action, disturbances
        return self.outputs(state)

    def constraint_costs(self, state, disturbances):
        del state, disturbances
        return {}

    def safety_margins(self, state, disturbances):
        del state, disturbances
        return {"state": 1.0}

    def step_info(self, state, action, disturbances):
        del action, disturbances
        return {"y": self.outputs(state)}

def _reward(state, action, next_state, context):
    del state, action
    error = float(context["reference"][0]) - float(next_state[0])
    return -(error**2), {"tracking": -(error**2)}


def register_toy() -> Scenario:
    unregister_scenario("core-toy")
    reward = Reward(
        id="regulation",
        function=_reward,
        episode_metric_function=lambda env, episode: {
            "return": episode.episode_return,
            "iae": sum(
                abs(
                    float(row.info["transition_reference"][0])
                    - float(row.info["y"][0])
                )
                for row in episode.transitions
            )
            * env.control_dt,
        },
        primary_metric="iae",
        metric_direction="minimize",
        safety_violation_penalty=5.0,
    )
    default_episode = EpisodeSpec(
        initial_state=(0.0,),
        initial_action=(0.0,),
        reference=(1.0,),
        horizon=3,
    )
    short_episode = EpisodeSpec(
        initial_state=(0.0,),
        initial_action=(0.25,),
        reference=(1.0,),
        horizon=2,
    )
    benchmarks = {
        name: Benchmark(
            id=name,
            reward_id="regulation",
            episode_factory=lambda model, rng, episode=short_episode: episode,
            metric_function=reward.episode_metric_function,
            ranking_metrics=(("iae", "minimize"),),
        )
        for name in ("tracking", "disturbance-rejection", "boundary-safety")
    }
    scenario = Scenario(
        id="core-toy",
        make_model=ToyModel,
        control_dt=0.5,
        make_default_episode=lambda model: default_episode,
        sample_training_episode=lambda model, rng, reward_id: (
            short_episode,
            "tracking",
        ),
        sample_training_disturbance=lambda model, rng: {},
        benchmarks=benchmarks,
        rewards={"regulation": reward},
        default_reward="regulation",
    )
    register_scenario(scenario)
    return scenario


def test_make_env_uses_scenario_model_and_concrete_environment():
    scenario = register_toy()
    try:
        env = make_env("core-toy", benchmark="boundary-safety")
        assert isinstance(env, ProcessControlEnv)
        observation, info = env.reset(seed=7)
        assert env.action_space.shape == (1,)
        assert env.observation_space.contains(observation)
        assert info["reward_id"] == "regulation"
        assert info["scenario_id"] == "core-toy"
        assert info["episode_parameters"] == {"case_seed": 7}
        assert info["previous_applied_action"] == pytest.approx([0.25])
        assert scenario.id == "core-toy"

        next_observation, reward, terminated, truncated, step_info = env.step(
            np.asarray([0.5], dtype=np.float32)
        )
        assert np.allclose(next_observation, [0.19661458])
        assert reward == pytest.approx(-(1.0 - float(next_observation[0])) ** 2)
        assert not terminated
        assert not truncated
        assert step_info["commanded_action"].tolist() == [0.5]
    finally:
        unregister_scenario("core-toy")


def test_benchmark_reset_requires_case_seed_and_rejects_episode_override():
    register_toy()
    env = None
    try:
        env = make_env("core-toy", benchmark="tracking")
        with pytest.raises(ValueError, match="requires a case seed"):
            env.reset()
        with pytest.raises(ValueError, match="cannot override"):
            env.reset(seed=0, options={"episode": env.unwrapped.default_episode})
    finally:
        if env is not None:
            env.close()
        unregister_scenario("core-toy")


def test_env_rejects_unknown_benchmark_and_non_space_action():
    register_toy()
    try:
        with pytest.raises(KeyError, match="unknown benchmark"):
            make_env("core-toy", benchmark="missing")
        env = make_env("core-toy")
        env.reset(seed=0)
        with pytest.raises(ValueError, match="action_space"):
            env.step(np.asarray([1.5], dtype=np.float32))
    finally:
        unregister_scenario("core-toy")


def test_constraints_are_computed_before_terminal_safety_penalty():
    register_toy()
    try:
        env = make_env("core-toy")
        env.model.constraint_costs = lambda state, disturbances: (
            {"state_limit": 1.0} if float(state[0]) > 0.3 else {}
        )
        env.model.safety_margins = lambda state, disturbances: {
            "state": 0.3 - float(state[0])
        }
        env.reset(seed=0)
        _, reward, terminated, _, info = env.step(
            np.asarray([1.0], dtype=np.float32)
        )
        base_reward = -(1.0 - float(info["y"][0])) ** 2
        assert terminated
        assert info["constraint_costs"] == {"state_limit": 1.0}
        assert info["reward_terms"]["safety"] == -5.0
        assert reward == pytest.approx(base_reward - 5.0)
    finally:
        unregister_scenario("core-toy")


@pytest.mark.parametrize(
    "variation", ("randomize", "disturbance", "noise", "delay", "fault")
)
def test_benchmark_rejects_training_variation(variation):
    register_toy()
    try:
        with pytest.raises(ValueError, match="benchmark cannot be combined"):
            make_env("core-toy", benchmark="tracking", **{variation: True})
    finally:
        unregister_scenario("core-toy")


def test_disturbance_switch_requires_boolean():
    with pytest.raises(TypeError, match="disturbance must be a boolean"):
        aiogym.make_env("quadruple", disturbance=1)


@pytest.mark.parametrize(
    "override",
    ({"reward": "regulation"}, {"parameters": {}}),
)
def test_benchmark_rejects_reward_and_parameter_overrides(override):
    register_toy()
    try:
        with pytest.raises(ValueError, match="fixes model parameters and reward"):
            make_env("core-toy", benchmark="tracking", **override)
    finally:
        unregister_scenario("core-toy")


@pytest.mark.parametrize("scenario", ("quadruple", "three_tank"))
def test_seeded_tracking_benchmark_is_reproducible_without_measurement_noise(
    scenario,
):
    env = aiogym.make_env(scenario, benchmark="tracking")
    try:
        first, first_info = env.reset(seed=7)
        repeated, repeated_info = env.reset(seed=7)
        different, different_info = env.reset(seed=8)
        assert np.array_equal(first, repeated)
        assert first_info["episode_spec"] == repeated_info["episode_spec"]
        assert not np.array_equal(first, different)
        assert first_info["episode_spec"] != different_info["episode_spec"]
        assert env.unwrapped.runtime_config["noise"] is None
    finally:
        env.close()


@pytest.mark.parametrize("scenario", ("quadruple", "three_tank"))
def test_observation_noise_does_not_change_reference_channels(scenario):
    clean = aiogym.make_env(scenario)
    noisy = aiogym.make_env(scenario, noise=True)
    try:
        clean_observation, _ = clean.reset(seed=17)
        noisy_observation, noisy_info = noisy.reset(seed=17)
        schema = tuple(noisy.unwrapped.model.observation_schema())
        reference_indices = np.asarray(
            [
                index
                for index, row in enumerate(schema)
                if "kind" in row and row["kind"] == "reference"
            ]
        )
        measurement_indices = np.asarray(
            [
                index
                for index, row in enumerate(schema)
                if "kind" in row and row["kind"] == "measurement"
            ]
        )

        assert reference_indices.size
        assert measurement_indices.size
        assert np.array_equal(
            noisy_observation[reference_indices],
            clean_observation[reference_indices],
        )
        assert not np.array_equal(
            noisy_observation[measurement_indices],
            clean_observation[measurement_indices],
        )
        bias = np.asarray(noisy_info["runtime_variation"]["observation_bias"])
        assert np.array_equal(bias[reference_indices], np.zeros(reference_indices.size))

        action = np.asarray(noisy.unwrapped.model.default_action(), dtype=np.float32)
        clean_next, *_ = clean.step(action)
        noisy_next, *_ = noisy.step(action)
        assert np.array_equal(
            noisy_next[reference_indices], clean_next[reference_indices]
        )
    finally:
        clean.close()
        noisy.close()

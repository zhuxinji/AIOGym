from __future__ import annotations

import numpy as np
import pytest

from aiogym.core.contracts import Scenario
from aiogym.core.env import make_env
from aiogym.core.registry import register_scenario, unregister_scenario
from aiogym.core.rollout import rollout
from aiogym.core.specs import Benchmark, EpisodeSpec, Reward
from aiogym.scenarios._metrics import regulation_episode_metrics
from aiogym.workflows import DatasetReader, collect, evaluate


REWARD_CONTEXTS = []


class ScheduleModel:
    scenario = "schedule-toy"
    dt_micro = 1.0

    def __init__(self, parameters=None):
        if parameters:
            raise ValueError("schedule toy has no parameters")
        self.resolved_parameters = {}

    def initial_state(self):
        return [0.0]

    def dynamics(self, state, action, disturbances=None):
        del state
        return [float(action[0]) * float(disturbances["gain"])]

    def outputs(self, state):
        return list(state)

    def action_schema(self):
        return [{"name": "u", "low": 0.0, "high": 1.0}]

    def state_schema(self):
        return [{"name": "x", "low": -100.0, "high": 100.0}]

    def observation_schema(self):
        return [
            {"name": "x", "low": -100.0, "high": 100.0},
            {"name": "reference", "low": 0.0, "high": 1.0},
            {"name": "gain", "low": 0.0, "high": 10.0},
        ]

    def observation(self, state, reference, previous_action, disturbances):
        del previous_action
        return [float(state[0]), float(reference[0]), float(disturbances["gain"])]

    def default_action(self):
        return [0.0]

    def default_setpoint_vector(self):
        return [0.1]

    def controlled_output_scales(self):
        return [1.0]

    def default_disturbances(self):
        return {"gain": 1.0}

    def output_schema(self):
        return [{"name": "x", "low": -100.0, "high": 100.0}]

    def action_slew_limits(self):
        return None

    def clamp_state(self, state):
        return state

    def constraint_costs(self, state, disturbances):
        del state, disturbances
        return {}

    def safety_margins(self, state, disturbances):
        del state, disturbances
        return {"state": 1.0}

    def step_info(self, state, action, disturbances):
        del action, disturbances
        return {"y": self.outputs(state), "energy_kw": 0.0}

class RecordingPolicy:
    env = None

    def __init__(self):
        self.calls = []

    def reset(self, seed=None):
        del seed
        self.calls = []

    def act(self, observation, context):
        info = context["info"]
        self.calls.append(
            {
                "observation": np.asarray(observation).copy(),
                "reference": np.asarray(context["reference"]).copy(),
                "disturbance": dict(info["disturbance"]),
                "step_index": context["step_index"],
                "physical_time": context["physical_time"],
            }
        )
        return np.asarray([context["reference"][0]], dtype=np.float32)

    def metadata(self):
        return {"id": "schedule-recorder"}


def _reward(state, action, next_state, context):
    del state, action
    REWARD_CONTEXTS.append(
        {
            "reference": list(context["reference"]),
            "disturbances": dict(context["disturbances"]),
            "step_index": context["step_index"],
        }
    )
    reward = float(context["reference"][0] - next_state[0])
    return reward, {"scheduled_reward": reward}


@pytest.fixture
def schedule_scenario():
    unregister_scenario("schedule-toy")
    REWARD_CONTEXTS.clear()
    reward = Reward(
        id="regulation",
        function=_reward,
        episode_metric_function=regulation_episode_metrics,
        primary_metric="return",
        metric_direction="maximize",
    )
    episode = EpisodeSpec(
        initial_state=(0.0,),
        initial_action=(0.0,),
        reference=(0.1,),
        horizon=3,
        disturbances={"gain": 2.0},
        reference_schedule={0: (0.2,), 1: (0.4,), 2: (0.6,)},
        disturbance_schedule={
            0: {"gain": 4.0},
            1: {"gain": 5.0},
            2: {"gain": 6.0},
        },
    )
    benchmarks = {
        name: Benchmark(
            id=name,
            reward_id="regulation",
            episode_factory=lambda model, rng, value=episode: value,
            metric_function=regulation_episode_metrics,
            ranking_metrics=(("tracking_iae", "minimize"),),
        )
        for name in ("tracking", "disturbance-rejection", "boundary-safety")
    }
    scenario = Scenario(
        id="schedule-toy",
        make_model=ScheduleModel,
        control_dt=1.0,
        make_default_episode=lambda model: episode,
        sample_training_episode=lambda model, rng, reward_id: (
            episode,
            "tracking",
        ),
        sample_training_disturbance=lambda model, rng: {},
        benchmarks=benchmarks,
        rewards={"regulation": reward},
        default_reward="regulation",
    )
    register_scenario(scenario)
    try:
        yield
    finally:
        unregister_scenario("schedule-toy")


def test_events_are_visible_before_corresponding_actions(schedule_scenario):
    env = make_env("schedule-toy")
    env.set_disturbances({"gain": 3.0})
    policy = RecordingPolicy()
    try:
        episode = rollout(env, policy, seed=7)
    finally:
        env.close()

    assert [call["reference"].tolist() for call in policy.calls] == [
        [0.2],
        [0.4],
        [0.6],
    ]
    assert [call["disturbance"]["gain"] for call in policy.calls] == [4.0, 5.0, 6.0]
    assert np.allclose(
        [call["observation"][1:] for call in policy.calls],
        [[0.2, 4.0], [0.4, 5.0], [0.6, 6.0]],
    )
    assert np.allclose(
        [row.action for row in episode.transitions],
        [[0.2], [0.4], [0.6]],
    )
    assert [row.step_index for row in episode.transitions] == [0, 1, 2]
    assert [row.physical_time for row in episode.transitions] == [1.0, 2.0, 3.0]
    assert [call["step_index"] for call in policy.calls] == [0, 1, 2]
    assert [call["physical_time"] for call in policy.calls] == [0.0, 1.0, 2.0]
    assert [row.info["transition_step_index"] for row in episode.transitions] == [
        0,
        1,
        2,
    ]
    assert [row.info["transition_reference"].tolist() for row in episode.transitions] == [
        [0.2],
        [0.4],
        [0.6],
    ]
    assert [row.info["reference"].tolist() for row in episode.transitions] == [
        [0.4],
        [0.6],
        [0.6],
    ]
    assert [row.info["transition_disturbance"]["gain"] for row in episode.transitions] == [
        4.0,
        5.0,
        6.0,
    ]
    assert [row.info["disturbance"]["gain"] for row in episode.transitions] == [
        5.0,
        6.0,
        6.0,
    ]
    assert REWARD_CONTEXTS == [
        {"reference": [0.2], "disturbances": {"gain": 4.0}, "step_index": 0},
        {"reference": [0.4], "disturbances": {"gain": 5.0}, "step_index": 1},
        {"reference": [0.6], "disturbances": {"gain": 6.0}, "step_index": 2},
    ]


def test_dataset_records_both_transition_and_next_contexts(
    schedule_scenario, tmp_path
):
    env = make_env("schedule-toy")
    try:
        result = collect(
            env=env,
            policy=RecordingPolicy(),
            episodes=1,
            output=tmp_path / "schedule",
            max_steps=3,
        )
    finally:
        env.close()
    reader = DatasetReader(result["path"])
    episode = reader.load_episode(0)
    assert reader.metadata["schema_version"] == "aiogym.dataset.v2"
    assert np.allclose(
        episode.array("transition_reference"), [[0.2], [0.4], [0.6]]
    )
    assert np.allclose(episode.array("reference"), [[0.4], [0.6], [0.6]])
    assert episode.array("transition_disturbance").tolist() == [
        [4.0],
        [5.0],
        [6.0],
    ]
    assert episode.array("disturbance").tolist() == [[5.0], [6.0], [6.0]]


def test_evaluation_metrics_use_the_same_transition_reference_as_reward(
    schedule_scenario,
):
    env = make_env("schedule-toy", reward="regulation")
    try:
        result = evaluate(
            env=env,
            policy=RecordingPolicy(),
            seeds=(0,),
            max_steps=3,
        )
    finally:
        env.close()
    assert result["schema_version"] == "aiogym.evaluation.v3"
    assert result["episodes"][0]["metrics"]["tracking_iae"] == pytest.approx(
        8.8
    )


def test_quadruple_schedule_changes_context_at_declared_physical_time():
    env = make_env("quadruple", benchmark="tracking")
    action = np.asarray(env.unwrapped.model.default_action(), dtype=np.float32)
    try:
        _, info = env.reset(seed=0)
        initial_reference = info["reference"].copy()
        scheduled_reference = env.unwrapped.episode.reference_schedule[120]
        for _ in range(120):
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated
            assert not truncated
        assert info["step_index"] == 120
        assert info["physical_time"] == 120.0
        assert info["transition_reference"] == pytest.approx(initial_reference)
        assert info["reference"] == pytest.approx(scheduled_reference)
        _, _, _, _, next_info = env.step(action)
        assert next_info["transition_reference"] == pytest.approx(info["reference"])
        assert next_info["transition_step_index"] == 120
    finally:
        env.close()

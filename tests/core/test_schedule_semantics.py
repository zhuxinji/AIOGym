from __future__ import annotations

import numpy as np
import pytest

from aiogym.core import (
    ResolvedPlant,
    ScenarioPlugin,
    TaskSpec,
    make_env,
    register_scenario,
    rollout,
    unregister_scenario,
)
from aiogym.workflows import DatasetReader, collect


REWARD_CONTEXTS = []


class ScheduleModel:
    scenario = "schedule-toy"

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

    def observation_schema(self, condition):
        del condition
        return [
            {"name": "x", "low": -100.0, "high": 100.0},
            {"name": "reference", "low": 0.0, "high": 1.0},
            {"name": "gain", "low": 0.0, "high": 10.0},
        ]

    def observation(self, state, reference, previous_action, disturbances, condition):
        del previous_action, condition
        return [float(state[0]), float(reference[0]), float(disturbances["gain"])]

    def default_action(self):
        return [0.0]

    def default_setpoint_vector(self):
        return [0.1]

    def default_disturbances(self):
        return {"gain": 1.0}


class RecordingPolicy:
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
def schedule_plugin():
    unregister_scenario("schedule-toy")
    REWARD_CONTEXTS.clear()
    task = TaskSpec(
        id="schedule-toy/regulation",
        scenario="schedule-toy",
        objective="regulation",
        reward=_reward,
        metrics=("return",),
        primary_metric="return",
        metric_direction="maximize",
    )
    plugin = ScenarioPlugin(
        id="schedule-toy",
        make_model=lambda plant: ScheduleModel(),
        default_plant=lambda: {
            "schema_version": "aiogym.plant.v2",
            "id": "schedule-plant",
            "scenario": "schedule-toy",
            "plant": {},
            "conditions": {
                "scheduled": {
                    "id": "scheduled",
                    "initial_state": [0.0],
                    "reference": [0.1],
                    "control_dt": 1.0,
                    "horizon": 3,
                    "disturbances": {"gain": 2.0},
                    "reference_schedule": {
                        0: [0.2],
                        1: [0.4],
                        2: [0.6],
                    },
                    "disturbance_schedule": {
                        0: {"gain": 4.0},
                        1: {"gain": 5.0},
                        2: {"gain": 6.0},
                    },
                }
            },
            "default_condition": "scheduled",
        },
        resolve_plant=lambda config: ResolvedPlant(config, config.plant),
        tasks={"regulation": task},
    )
    register_scenario(plugin)
    try:
        yield
    finally:
        unregister_scenario("schedule-toy")


def test_events_are_visible_before_corresponding_actions(schedule_plugin):
    env = make_env("schedule-toy/regulation")
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


def test_dataset_v4_records_both_transition_and_next_contexts(
    schedule_plugin, tmp_path
):
    result = collect(
        task="schedule-toy/regulation",
        policy=RecordingPolicy(),
        episodes=1,
        output=tmp_path / "schedule-v4",
        max_steps=3,
    )
    reader = DatasetReader(result["path"], verify_checksums=True)
    episode = reader.load_episode(0)
    assert reader.manifest["schema_version"] == "aiogym.dataset.v4"
    assert reader.manifest["schedule_semantics"] == "pre-action-v1"
    assert episode.metadata["schema_version"] == "aiogym.dataset.episode.v4"
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


def test_quadruple_schedule_changes_context_at_declared_physical_time():
    env = make_env("quadruple/regulation", condition="minimum-phase")
    action = np.asarray(env.model.default_action(), dtype=np.float32)
    try:
        _, info = env.reset(seed=0)
        initial_reference = info["reference"].copy()
        for _ in range(120):
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated
            assert not truncated
        assert info["step_index"] == 120
        assert info["physical_time"] == 120.0
        assert info["transition_reference"] == pytest.approx(initial_reference)
        assert info["reference"] == pytest.approx(
            [13.2629675195507, 11.783158403008972]
        )
        _, _, _, _, next_info = env.step(action)
        assert next_info["transition_reference"] == pytest.approx(info["reference"])
        assert next_info["transition_step_index"] == 120
    finally:
        env.close()

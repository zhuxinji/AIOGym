"""Contracts for model-based transition data collection."""
from __future__ import annotations

import json
import math
import unittest

import numpy as np

from aiogym.env import AIOGymEnv
from aiogym.controllers import make_controller
from aiogym.evaluation import rollout_controller
from aiogym.rl import Transition, TransitionDataset, collect_transitions


def _environment(*, action_mode="actuator"):
    return AIOGymEnv(
        "cstr",
        reward_spec="regulation-v1",
        action_mode=action_mode,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
        episode_steps=3,
    )


class TransitionDatasetTests(unittest.TestCase):
    def test_random_collection_is_reproducible_and_episode_aware(self):
        first = collect_transitions(_environment(), episodes=2, seed=17)
        second = collect_transitions(_environment(), episodes=2, seed=17)

        self.assertEqual(len(first), 6)
        self.assertEqual(first.to_rows(), second.to_rows())
        self.assertEqual([item.episode for item in first], [0, 0, 0, 1, 1, 1])
        self.assertEqual([item.step for item in first], [0, 1, 2, 0, 1, 2])
        self.assertEqual([item.truncated for item in first], [False, False, True] * 2)
        self.assertTrue(all(not item.terminated for item in first))
        self.assertTrue(all(item.setpoint for item in first))
        self.assertTrue(all(item.disturbance for item in first))

    def test_supervised_and_rl_views_preserve_dimensions(self):
        dataset = collect_transitions(_environment(), episodes=1, seed=3)
        states, actions, next_states = dataset.supervised_arrays("state")
        observations, observation_actions, next_observations = dataset.supervised_arrays("observation")

        self.assertEqual(states.shape, (3, dataset.state_dim))
        self.assertEqual(actions.shape, (3, dataset.action_dim))
        self.assertEqual(next_states.shape, states.shape)
        self.assertEqual(observations.shape, (3, dataset.observation_dim))
        self.assertEqual(observation_actions.shape, actions.shape)
        self.assertEqual(next_observations.shape, observations.shape)
        self.assertTrue(np.all(np.isfinite(states)))
        self.assertTrue(np.all(np.isfinite(observations)))

    def test_rows_round_trip_and_existing_dataset_append(self):
        dataset = collect_transitions(_environment(), episodes=1, seed=8)
        json.dumps(dataset.to_rows())
        restored = TransitionDataset.from_rows(dataset.to_rows())
        self.assertEqual(restored.to_rows(), dataset.to_rows())

        collect_transitions(_environment(), episodes=1, seed=9, dataset=restored)
        self.assertEqual(len(restored), 6)
        self.assertEqual([item.episode for item in restored], [0, 0, 0, 1, 1, 1])

    def test_existing_rollout_artifact_converts_to_transition_dataset(self):
        env = _environment()
        artifact = rollout_controller(
            make_controller("pid", scenario="cstr"),
            env,
            seed=12,
            max_steps=2,
        )
        dataset = TransitionDataset.from_rollout(artifact, episode=4)

        self.assertEqual(len(dataset), 2)
        self.assertEqual([item.episode for item in dataset], [4, 4])
        self.assertEqual(dataset[0].setpoint, tuple(artifact["rollout"][0]["setpoint"]["y_sp"]))
        self.assertEqual(dataset[0].state, tuple(artifact["rollout"][0]["state"]))

    def test_controller_collection_uses_controller_actions(self):
        class ConstantPolicy:
            def act(self, obs, context):
                return np.array([0.25, 0.75], dtype=np.float32)

        dataset = collect_transitions(_environment(), ConstantPolicy(), episodes=1, seed=4)
        for transition in dataset:
            self.assertEqual(transition.action, (0.25, 0.75))
            self.assertTrue(math.isfinite(transition.reward))

    def test_invalid_transitions_and_collection_modes_fail_early(self):
        valid = Transition(
            obs=[0.0, 1.0],
            state=[0.0],
            action=[0.5],
            reward=0.0,
            next_obs=[0.1, 1.0],
            next_state=[0.1],
            terminated=False,
            truncated=False,
        )
        dataset = TransitionDataset([valid])
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            dataset.append(Transition(
                obs=[0.0, 1.0, 2.0],
                state=[0.0],
                action=[0.5],
                reward=0.0,
                next_obs=[0.1, 1.0, 2.0],
                next_state=[0.1],
                terminated=False,
                truncated=False,
            ))
        with self.assertRaisesRegex(ValueError, "reward must be finite"):
            Transition(
                obs=[0.0],
                state=[0.0],
                action=[0.5],
                reward=float("nan"),
                next_obs=[0.1],
                next_state=[0.1],
                terminated=False,
                truncated=False,
            )
        with self.assertRaisesRegex(ValueError, "action_mode='actuator'"):
            collect_transitions(_environment(action_mode="setpoint"), episodes=1)

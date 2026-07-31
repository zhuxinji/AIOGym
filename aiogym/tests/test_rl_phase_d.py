"""Phase-D acceptance tests for the unified online-training substrate."""
from __future__ import annotations

import random

import numpy as np
import pytest
import gymnasium as gym

from aiogym.tests._env import make_test_env as make_env
from aiogym.rl.backends.sb3_algorithms import list_algorithm_adapters
from aiogym.rl.checkpoints import (
    CheckpointManager,
    TrainingCheckpoint,
    capture_rng_state,
    restore_rng_state,
)
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.coordinator import EpisodeCoordinator
from aiogym.rl.replay import ReplayBuffer
from aiogym.rl.statistics import (
    NormalizedActionWrapper,
    ObservationNormalizationWrapper,
    RunningObservationStatistics,
)
from aiogym.rl.utd import UTDController, sb3_update_schedule
from aiogym.rl.vector import vector_transition_batch


def _config(*, n_envs=1):
    return RLTrainingConfig(
        track_id="phase-d-smoke",
        algorithm_id="sac",
        training_seed=17,
        total_transitions=20,
        n_envs=n_envs,
        algorithm={"utd_ratio": 0.5, "batch_size": 2},
        replay={"capacity": 32},
    )


def test_unified_config_has_stable_canonical_hash():
    first = _config(n_envs=4)
    second = RLTrainingConfig.from_mapping(first.as_dict())
    assert first.canonical_json() == second.canonical_json()
    assert first.config_hash == second.config_hash
    assert list_algorithm_adapters() == (
        "lagrangian_sac",
        "ppo",
        "sac",
        "td3",
    )


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0, 2.0])
def test_utd_is_invariant_to_n_envs(ratio):
    totals = []
    for n_envs in (1, 2, 4, 8):
        controller = UTDController(ratio)
        updates = 0
        remaining = 32
        while remaining:
            batch = min(n_envs, remaining)
            updates += controller.observe(batch)
            remaining -= batch
        totals.append(updates)
        vector_steps, gradient_steps = sb3_update_schedule(
            utd_ratio=ratio,
            n_envs=n_envs,
        )
        assert gradient_steps / (vector_steps * n_envs) == pytest.approx(ratio)
    assert len(set(totals)) == 1


def test_coordinator_sequence_is_invariant_to_worker_count():
    sequences = []
    for n_envs in (1, 3, 8):
        coordinator = EpisodeCoordinator(
            base_seed=12,
            namespace="phase-d-training",
        )
        assignments = coordinator.claim_many(20, n_envs=n_envs)
        sequences.append(
            [(row.episode_index, row.episode_seed) for row in assignments]
        )
        assert len({row.episode_seed for row in assignments}) == 20
    assert sequences[0] == sequences[1] == sequences[2]


def test_vector_terminal_observation_is_stored():
    batch = vector_transition_batch(
        observation=np.asarray([[1.0], [2.0]]),
        action=np.asarray([[0.1], [0.2]]),
        reward=np.asarray([1.0, 2.0]),
        next_observation=np.asarray([[100.0], [3.0]]),
        terminated=np.asarray([True, False]),
        truncated=np.asarray([False, False]),
        infos=[
            {"terminal_observation": np.asarray([9.0])},
            {},
        ],
    )
    assert batch.next_observation[:, 0].tolist() == [9.0, 3.0]
    assert batch.reset_observation[:, 0].tolist() == [100.0, 3.0]
    assert batch.bootstrap_mask.tolist() == [0.0, 1.0]


def test_action_normalization_round_trip_and_info_contract():
    env = make_env("cstr", episode_steps=1, auto_events=False)
    wrapped = NormalizedActionWrapper(env)
    try:
        physical = np.full(env.action_space.shape, 0.23, dtype=np.float32)
        normalized = wrapped.normalize(physical)
        assert wrapped.denormalize(normalized) == pytest.approx(physical)
        wrapped.reset(seed=3)
        _, _, _, _, info = wrapped.step(normalized)
        assert info["action_policy_normalized"] == pytest.approx(normalized)
        assert info["action_commanded_physical"] == pytest.approx(physical)
        assert np.all(wrapped.action_space.low == -1.0)
        assert np.all(wrapped.action_space.high == 1.0)
    finally:
        wrapped.close()


def test_validation_env_does_not_update_normalizer():
    statistics = RunningObservationStatistics((2,))
    train_env = ObservationNormalizationWrapper(
        _ObservationEnv(),
        statistics,
        training=True,
    )
    validation_env = ObservationNormalizationWrapper(
        _ObservationEnv(),
        statistics,
        training=False,
    )
    train_env.reset()
    train_env.step(np.asarray([0.0]))
    before = statistics.state_dict()
    validation_env.reset()
    validation_env.step(np.asarray([0.0]))
    after = statistics.state_dict()
    assert after["count"] == before["count"]
    assert after["mean"] == pytest.approx(before["mean"])
    assert after["m2"] == pytest.approx(before["m2"])


def test_minimal_info_keeps_trainer_channels_only():
    env = make_env(
        "cstr",
        episode_steps=1,
        auto_events=False,
        info_level="minimal",
    )
    try:
        env.reset(seed=4)
        _, _, _, _, info = env.step(env.action_space.sample())
    finally:
        env.close()
    assert {"reward_terms", "costs"} <= set(info)
    assert "y" not in info
    assert "temps" not in info


def test_resume_matches_uninterrupted_training_smoke(tmp_path):
    expected = _run_deterministic_smoke(12)

    random.seed(91)
    np.random.seed(91)
    config = _config()
    replay = ReplayBuffer(32, seed=19)
    utd = UTDController(config.utd_ratio)
    value = _advance_smoke(0.0, replay, utd, 5)
    checkpoint = TrainingCheckpoint(
        config=config,
        transition_count=5,
        update_count=utd.updates,
        algorithm_state={"value": value, "utd": utd.state_dict()},
        replay_state=replay.state_dict(),
        normalization_state=None,
        coordinator_state=EpisodeCoordinator(
            base_seed=17,
            namespace="smoke",
            next_episode_index=5,
        ).state_dict(),
        curriculum_state={"level": "L0"},
        best_validation={"score": 1.0},
        rng_state=capture_rng_state(),
        code_commit="test",
    )
    path = CheckpointManager.save(tmp_path / "resume.ckpt", checkpoint)

    restored = CheckpointManager.load(path, expected_config=config)
    restore_rng_state(restored.rng_state)
    replay = ReplayBuffer.from_state_dict(restored.replay_state)
    utd = UTDController.from_state_dict(restored.algorithm_state["utd"])
    resumed = _advance_smoke(
        restored.algorithm_state["value"],
        replay,
        utd,
        7,
    )
    assert resumed == pytest.approx(expected)
    assert utd.transitions == 12
    assert len(replay) == 12


def _run_deterministic_smoke(steps):
    random.seed(91)
    np.random.seed(91)
    replay = ReplayBuffer(32, seed=19)
    utd = UTDController(0.5)
    return _advance_smoke(0.0, replay, utd, steps)


def _advance_smoke(value, replay, utd, steps):
    for _ in range(steps):
        observation = np.asarray([np.random.random()], dtype=np.float32)
        reward = random.random()
        replay.add(
            observation=observation,
            action=np.asarray([0.0]),
            reward=reward,
            next_observation=observation + 1.0,
            terminated=False,
            truncated=False,
        )
        for _ in range(utd.observe(1)):
            value += float(replay.sample(1)["reward"][0])
    return value


class _ObservationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            (1,),
            dtype=np.float32,
        )
        self._value = 0.0

    def reset(self, *, seed=None, options=None):
        self._value = 1.0
        return np.asarray([self._value, 2.0], dtype=np.float32), {}

    def step(self, action):
        self._value += 1.0
        return (
            np.asarray([self._value, 2.0], dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )

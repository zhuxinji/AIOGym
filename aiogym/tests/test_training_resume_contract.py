from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiogym.benchmarks import load_track
from aiogym.rl.checkpoints import (
    CheckpointManager,
    TrainingCheckpoint,
    validate_resume_config,
    validate_resume_mode,
)
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.dataset_replay import DatasetReplay
from aiogym.rl.online_collection import VectorOnlineCollector
from aiogym.rl.replay import ReplayBuffer
from aiogym.rl.episode_env import (
    make_track_episode_sampler,
    make_track_training_base_env,
)
from aiogym.rl.train_sb3 import _capture_sb3_resume_state


TRACK_ID = "quadruple-regulation-generalist-v1"


def _config(*, n_envs=2, track_id=TRACK_ID):
    return RLTrainingConfig(
        track_id=track_id,
        algorithm_id="sac",
        training_seed=3,
        total_transitions=100,
        n_envs=n_envs,
        algorithm={"utd_ratio": 1.0, "rollout_vector_steps": n_envs},
        replay={"schema": "v1"},
    )


def test_restart_episode_uses_real_coordinator_and_avoids_duplicates():
    track = load_track(TRACK_ID)
    sampler = make_track_episode_sampler(track)

    def factories(count):
        return [
            lambda: make_track_training_base_env(
                track,
                sampler=sampler,
            )
            for _ in range(count)
        ]

    first = VectorOnlineCollector(
        factories(2),
        ReplayBuffer(32, seed=1),
        base_seed=7,
        namespace=track.seed_namespace("training"),
        sampler=sampler,
        track=track,
    )
    try:
        state = first.resume_state()
        assert state["active_episode_indexes"] == [0, 1]
        assert state["next_episode_index"] == 2
        assert state["partial_episodes_discarded"] == 2
    finally:
        first.close()

    resumed = VectorOnlineCollector(
        factories(3),
        ReplayBuffer(32, seed=1),
        base_seed=7,
        namespace=track.seed_namespace("training"),
        sampler=sampler,
        track=track,
        coordinator_state=state,
    )
    try:
        resumed_state = resumed.resume_state()
        assert resumed_state["active_episode_indexes"] == [2, 3, 4]
        assert not (
            set(state["active_episode_indexes"])
            & set(resumed_state["active_episode_indexes"])
        )
    finally:
        resumed.close()


def test_training_checkpoint_records_restart_contract_round_trip(
    tmp_path,
):
    config = _config()
    coordinator = {
        "resume_mode": "restart_episode",
        "next_episode_index": 9,
    }
    checkpoint = TrainingCheckpoint(
        config=config,
        transition_count=12,
        update_count=4,
        algorithm_state={"kind": "test"},
        replay_state={"schema": "v1"},
        normalization_state=None,
        coordinator_state=coordinator,
        curriculum_state=None,
        best_validation=None,
        rng_state={},
        resume_mode="restart_episode",
        last_committed_episode_index=6,
        next_episode_index=9,
        partial_episodes_discarded=2,
        n_envs=2,
        vector_backend="subproc",
    )
    path = tmp_path / "checkpoint.pkl"
    CheckpointManager.save(path, checkpoint)
    restored = CheckpointManager.load(path, expected_config=config)
    assert restored.coordinator_state == coordinator
    assert restored.next_episode_index == 9
    assert restored.partial_episodes_discarded == 2
    assert restored.resume_mode == "restart_episode"


def test_resume_allows_env_count_but_rejects_identity_changes():
    validate_resume_config(_config(n_envs=1), _config(n_envs=4))
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_config(
            _config(),
            _config(track_id="cascade-regulation-generalist-v1"),
        )
    with pytest.raises(ValueError, match="complete environment"):
        validate_resume_mode("exact_single_process", "subproc")


def test_sb3_checkpoint_captures_env_reported_state_not_timestep_math():
    track = load_track(TRACK_ID)
    worker_states = [
        {
            "coordinator": {"next_episode_index": 9},
            "active_episode_index": 7,
            "last_completed_episode_index": 5,
            "worker_index": 0,
        },
        {
            "coordinator": {"next_episode_index": 10},
            "active_episode_index": 8,
            "last_completed_episode_index": 6,
            "worker_index": 1,
        },
    ]

    class VectorEnv:
        def env_method(self, name):
            assert name == "training_resume_state"
            return worker_states

    class Model:
        num_timesteps = 999_999

        def get_env(self):
            return VectorEnv()

    args = SimpleNamespace(
        track_spec=track,
        seed=3,
        training_seed_namespace=track.seed_namespace("training"),
        n_envs=2,
        vec_env="subproc",
        resolved_reward_spec_id=track.reward_spec_id,
        algo="sac",
    )
    state = _capture_sb3_resume_state(args, Model())
    assert state["next_episode_index"] == 9
    assert state["last_committed_episode_index"] == 6
    assert state["partial_episodes_discarded"] == 2

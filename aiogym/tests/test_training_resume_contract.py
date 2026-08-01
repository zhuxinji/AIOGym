from __future__ import annotations

from dataclasses import replace

import pytest

from aiogym.benchmarks import load_track
from aiogym.rl.checkpoints import (
    CheckpointManager,
    LEGACY_TRAINING_CHECKPOINT_SCHEMA_VERSION,
    TrainingCheckpoint,
    selected_checkpoint_manifest,
    validate_resume_config,
)
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.online_collection import VectorOnlineCollector
from aiogym.rl.replay import ReplayBuffer
from aiogym.rl.episode_env import (
    make_track_episode_sampler,
    make_track_training_base_env,
)
from aiogym.rl.backends.sb3 import _capture_sb3_resume_state
from aiogym.rl.plan import resolve_training_plan


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
    with pytest.raises(ValueError, match="restart_episode"):
        replace(checkpoint, resume_mode="exact_single_process")


def test_resume_rejects_changed_n_envs_and_identity_changes():
    with pytest.raises(ValueError, match="unchanged n_envs"):
        validate_resume_config(_config(n_envs=1), _config(n_envs=4))
    validate_resume_config(
        _config(),
        replace(
            _config(),
            total_transitions=200,
            resume_checkpoint="/tmp/policy.zip",
        ),
    )
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_config(
            _config(),
            _config(track_id="cascade-regulation-generalist-v1"),
        )


def test_resume_rejects_changed_n_envs_for_rlpd():
    first = RLTrainingConfig(
        track_id=TRACK_ID,
        algorithm_id="rlpd",
        training_seed=3,
        total_transitions=100,
        n_envs=1,
        dataset_id="dataset",
        dataset_path="dataset",
    )
    with pytest.raises(ValueError, match="unchanged n_envs"):
        validate_resume_config(first, replace(first, n_envs=2))


def test_tampered_best_checkpoint_hash_is_rejected(tmp_path):
    policy = tmp_path / "best.zip"
    policy.write_bytes(b"canonical-best")
    record = {"checkpoint_id": "step-10", "step": 10}
    manifest = selected_checkpoint_manifest(policy, record)
    validation_state = {
        "schema_version": "test",
    }
    checkpoint = TrainingCheckpoint(
        config=_config(),
        transition_count=10,
        update_count=1,
        algorithm_state={},
        replay_state=None,
        normalization_state=None,
        coordinator_state={},
        curriculum_state=None,
        best_validation=None,
        rng_state={},
        validation_state=validation_state,
        selected_checkpoint=manifest,
    )
    policy.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        checkpoint.require_exact_validation_resume()


def test_legacy_checkpoint_is_not_claimed_as_exact_resume():
    config = _config()
    payload = TrainingCheckpoint(
        config=config,
        transition_count=10,
        update_count=1,
        algorithm_state={},
        replay_state=None,
        normalization_state=None,
        coordinator_state={},
        curriculum_state=None,
        best_validation={"step": 10},
        rng_state={},
    ).payload()
    payload["schema_version"] = LEGACY_TRAINING_CHECKPOINT_SCHEMA_VERSION
    payload.pop("validation_state")
    payload.pop("selected_checkpoint")
    payload.pop("legacy_partial_validation_state")
    restored = TrainingCheckpoint.from_payload(payload)
    assert restored.legacy_partial_validation_state is True
    assert restored.best_validation == {"step": 10}
    with pytest.raises(ValueError, match="legacy v1"):
        restored.require_exact_validation_resume()
def test_rlpd_rejects_fractional_utd_ratio():
    with pytest.raises(ValueError, match="positive integer"):
        RLTrainingConfig(
            track_id=TRACK_ID,
            algorithm_id="rlpd",
            training_seed=0,
            total_transitions=1,
            n_envs=1,
            algorithm={"utd_ratio": 1.5},
        )


@pytest.mark.parametrize("utd_ratio", [True, "1"])
def test_rlpd_rejects_non_numeric_utd_ratio(utd_ratio):
    with pytest.raises(TypeError, match="must be a number"):
        RLTrainingConfig(
            track_id=TRACK_ID,
            algorithm_id="rlpd",
            training_seed=0,
            total_transitions=1,
            n_envs=1,
            algorithm={"utd_ratio": utd_ratio},
        )


def test_training_config_rejects_removed_exact_resume_mode():
    with pytest.raises(ValueError, match="restart_episode"):
        replace(
            _config(n_envs=1),
            resume_mode="exact_single_process",
        )


def test_sb3_checkpoint_captures_env_reported_state_not_timestep_math():
    plan = resolve_training_plan(_config())
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

    state = _capture_sb3_resume_state(plan, Model())
    assert state["next_episode_index"] == 9
    assert state["last_committed_episode_index"] == 6
    assert state["partial_episodes_discarded"] == 2

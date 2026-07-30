"""Phase-E acceptance tests for Dataset replay and RLPD v2."""
from __future__ import annotations

import numpy as np
import pytest

from aiogym.datasets import DatasetEpisode, DatasetWriter
from aiogym.rl.behavior_cloning import BehaviorCloningTrainer
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.dataset_replay import DatasetReplay
from aiogym.rl.hybrid_replay import RLPDBatchSampler
from aiogym.rl.replay import ReplayBuffer
from aiogym.rl.rlpd import RLPD
from aiogym.rl.training_artifacts import rl_payload
from aiogym.rl.online_collection import VectorOnlineCollector
from aiogym.tests._env import make_test_env as make_env


torch = pytest.importorskip("torch")


def test_bc_reproduces_collector_on_tiny_dataset(tmp_path):
    path = _dataset(tmp_path, episode_count=1, constant_action=0.35)
    replay = DatasetReplay(path, seed=3)
    trainer = BehaviorCloningTrainer(
        replay,
        hidden=16,
        learning_rate=1e-2,
        seed=4,
    )
    report = trainer.fit(steps=200, batch_size=16)
    episode = replay.reader.load_episode(0)
    prediction = trainer.policy.normalized_action(
        episode.array("observation")
    )
    target = episode.array("action_policy_normalized")
    assert np.mean((prediction - target) ** 2) < 1e-3
    assert report["final_mse"] < report["initial_mse"]


def test_rlpd_batch_is_half_offline_half_online_in_canonical_mode(tmp_path):
    offline = DatasetReplay(_dataset(tmp_path), seed=7)
    online = ReplayBuffer(32, seed=8)
    for index in range(8):
        _add_online(online, float(index), terminated=False, truncated=False)
    sampler = RLPDBatchSampler(offline, online)
    batch = sampler.sample(10)
    assert np.sum(batch["source"] == "offline") == 5
    assert np.sum(batch["source"] == "online") == 5
    assert sampler.accounting()["sampled_offline_fraction"] == 0.5


def test_source_stratified_sampling_covers_small_collectors(tmp_path):
    replay = DatasetReplay(_dataset(tmp_path), seed=11, stratify=True)
    batch = replay.sample(200)
    assert set(batch["collector_id"]) == {"mpc", "nominal_pid"}
    assert set(replay.strata) == {"mpc|L1", "nominal_pid|L0"}


def test_rlpd_timeout_bootstrap_semantics(tmp_path):
    offline = DatasetReplay(_dataset(tmp_path), seed=12)
    online = ReplayBuffer(16, seed=13)
    _add_online(online, 1.0, terminated=False, truncated=True)
    _add_online(online, 2.0, terminated=True, truncated=False)
    sampler = RLPDBatchSampler(
        offline,
        online,
        offline_fraction=0.0,
        canonical=False,
    )
    batch = sampler.sample(100)
    timeout = np.logical_and(batch["truncated"], ~batch["terminated"])
    terminal = batch["terminated"]
    assert np.all(batch["bootstrap_mask"][timeout] == 1.0)
    assert np.all(batch["bootstrap_mask"][terminal] == 0.0)


def test_rlpd_resume_restores_buffers_and_rng(tmp_path):
    path = _dataset(tmp_path)
    agent = RLPD(
        2,
        1,
        hidden=16,
        n_critics=2,
        subset=1,
        batch=4,
        online_capacity=32,
        seed=21,
    )
    agent.load_dataset(path)
    for index in range(6):
        agent.push(
            np.asarray([index, -index], dtype=np.float32),
            np.asarray([0.1], dtype=np.float32),
            float(index),
            np.asarray([index + 1, -index], dtype=np.float32),
            False,
            truncated=index == 5,
        )
    agent.update()
    state = agent.state_dict()
    expected_batch = agent.sample_batch()
    expected_action = agent.policy_action_batch(
        np.asarray([[0.2, -0.3]], dtype=np.float32)
    )

    restored = RLPD(
        2,
        1,
        hidden=16,
        n_critics=2,
        subset=1,
        batch=4,
        online_capacity=32,
        seed=999,
    )
    restored.load_state_dict(state)
    actual_batch = restored.sample_batch()
    actual_action = restored.policy_action_batch(
        np.asarray([[0.2, -0.3]], dtype=np.float32)
    )
    for field in (
        "observation",
        "action",
        "reward",
        "bootstrap_mask",
        "source",
    ):
        assert np.array_equal(actual_batch[field], expected_batch[field])
    assert actual_action == pytest.approx(expected_action)
    assert len(restored.online) == len(agent.online)
    assert restored.gradient_updates == agent.gradient_updates


def test_dataset_id_and_hash_enter_artifact(tmp_path):
    replay = DatasetReplay(_dataset(tmp_path), seed=5)
    config = RLTrainingConfig(
        track_id="custom",
        algorithm_id="rlpd",
        training_seed=1,
        total_transitions=10,
        n_envs=2,
        algorithm={"utd_ratio": 5.0},
        dataset_id=replay.dataset_id,
    )
    training = {
        "algo": "rlpd",
        "dataset_id": replay.dataset_id,
        "dataset_hash": replay.dataset_hash,
        "training_config_hash": config.config_hash,
    }
    artifact = rl_payload(
        kind="rlpd_train_eval",
        scenario="quadruple",
        goal="regulation",
        action_mode="actuator",
        training=training,
        evaluation={},
        results=[],
        rows=[],
    )
    assert artifact["training"]["dataset_id"] == replay.dataset_id
    assert artifact["training"]["dataset_hash"] == replay.dataset_hash
    assert artifact["training"]["training_config_hash"] == config.config_hash


def test_vectorized_online_collection_accounts_for_timeouts():
    replay = ReplayBuffer(16, seed=30)
    collector = VectorOnlineCollector(
        [
            lambda: make_env(
                "cstr",
                episode_steps=1,
                auto_events=False,
                info_level="minimal",
            )
            for _ in range(2)
        ],
        replay,
        base_seed=31,
        namespace="phase-e-online",
    )
    try:
        report = collector.collect(
            lambda observations: np.zeros(
                (len(observations), 2),
                dtype=np.float32,
            ),
            5,
        )
    finally:
        collector.close()
    state = replay.state_dict()
    assert report["online_transitions"] == 5
    assert report["completed_episodes"] == 5
    assert len(replay) == 5
    assert np.all(state["arrays"]["truncated"][:5])
    assert np.all(state["arrays"]["bootstrap_mask"][:5] == 1.0)


def _dataset(tmp_path, *, episode_count=2, constant_action=None):
    path = tmp_path / "dataset"
    with DatasetWriter(
        path,
        dataset_id="phase-e-prior-v2",
        split="training",
    ) as writer:
        for index in range(episode_count):
            writer.append_episode(
                _episode(index, constant_action=constant_action)
            )
    return path


def _episode(index, *, constant_action=None):
    count = 16
    rng = np.random.default_rng(100 + index)
    observation = rng.normal(size=(count, 2)).astype(np.float32)
    if constant_action is None:
        action = np.tanh(
            0.4 * observation[:, :1] - 0.2 * observation[:, 1:]
        ).astype(np.float32)
    else:
        action = np.full((count, 1), constant_action, dtype=np.float32)
    terminated = np.zeros(count, dtype=np.bool_)
    truncated = np.zeros(count, dtype=np.bool_)
    truncated[-1] = True
    metadata = {
        "episode_id": f"phase-e-{index}",
        "split": "training",
        "track_id": "phase-e-track",
        "distribution_id": "phase-e-distribution",
        "distribution_hash": f"distribution-{index}",
        "episode_spec_id": f"spec-{index}",
        "resolved_hash": f"resolved-{index}",
        "base_seed": index,
        "component_seeds": {"policy": index},
        "scenario": "quadruple",
        "goal": "regulation",
        "action_mode": "actuator",
        "collector_id": "nominal_pid" if index == 0 else "mpc",
        "policy_id": "collector-policy",
        "collector_quality_tag": "expert",
        "plant_parameters": {},
        "initial_state": observation[0].tolist(),
        "reference_schedule": [{"at_step": 0, "values": [0.0]}],
        "disturbance_schedule": [],
        "sensor_model": {"kind": "none"},
        "actuator_model": {"kind": "none"},
        "termination_reason": "time_limit",
        "difficulty_tags": [f"L{index}"],
        "summary": {},
    }
    return DatasetEpisode(
        metadata=metadata,
        observation=observation,
        true_state=observation,
        reference=np.zeros((count, 1), dtype=np.float32),
        measured_disturbance=np.empty((count, 0), dtype=np.float32),
        action_policy_normalized=action,
        action_commanded_physical=0.5 * (action + 1.0),
        action_applied_physical=0.5 * (action + 1.0),
        reward_scalar=np.zeros(count, dtype=np.float32),
        reward_terms={"tracking": np.zeros(count, dtype=np.float32)},
        cost_channels={"safety": np.zeros(count, dtype=np.float32)},
        next_observation=observation + 0.01,
        next_true_state=observation + 0.01,
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=np.ones(count, dtype=np.float32),
        step_index=np.arange(count),
        physical_time=np.arange(1, count + 1, dtype=np.float32),
    )


def _add_online(replay, value, *, terminated, truncated):
    replay.add(
        observation=np.asarray([value, -value], dtype=np.float32),
        action=np.asarray([0.0], dtype=np.float32),
        reward=value,
        next_observation=np.asarray([value + 1.0, -value], dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
    )

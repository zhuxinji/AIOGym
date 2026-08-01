"""Phase-E acceptance tests for Dataset replay and RLPD v2."""
from __future__ import annotations

import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.rl

from aiogym import load_track

from aiogym.datasets.schema import DatasetEpisode
from aiogym.datasets.writer import DatasetWriter
from aiogym.rl.behavior_cloning import (
    BehaviorCloningPolicy,
    BehaviorCloningTrainer,
)
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.dataset_replay import DatasetReplay
from aiogym.rl.hybrid_replay import RLPDBatchSampler
from aiogym.rl.replay import ReplayBuffer
from aiogym.rl.policy_spec import (
    POLICY_SPEC_SCHEMA_VERSION,
    validate_policy_spec,
)
from aiogym.rl.rlpd import Actor, RLPD
from aiogym.rl.training_artifacts import rl_payload
from aiogym.rl.backends.bc import run_bc
from aiogym.rl.plan import resolve_training_plan
from aiogym.rl.runner import run_experiment
from aiogym.controllers.export import ExportResult
from aiogym.rl.online_collection import VectorOnlineCollector
from aiogym.rewards import get_reward_spec
from aiogym.tests._env import make_test_env as make_env


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


def test_bc_checkpoint_is_self_describing_and_resume_compatible(
    tmp_path,
):
    dataset_path = _dataset(
        tmp_path,
        episode_count=1,
        constant_action=0.25,
    )
    trainer = BehaviorCloningTrainer(
        DatasetReplay(dataset_path, seed=3),
        hidden=16,
        learning_rate=1e-2,
        seed=4,
    )
    report = trainer.fit(steps=2, batch_size=8)
    checkpoint = trainer.checkpoint_payload(report)
    path = tmp_path / "bc.pt"
    torch.save(checkpoint, path)
    restored_payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    assert set(restored_payload) == {
        "policy_spec",
        "policy_state_dict",
        "trainer",
        "report",
    }
    spec = validate_policy_spec(restored_payload["policy_spec"])
    assert spec["schema_version"] == POLICY_SPEC_SCHEMA_VERSION
    assert spec["algorithm_id"] == "bc"
    assert spec["network"]["hidden_sizes"] == [16, 16]
    assert spec["action_contract"]["native"] == "normalized[-1,1]"
    assert "optimizer" not in restored_payload["policy_state_dict"]
    assert "dataset" not in restored_payload["policy_state_dict"]

    policy = BehaviorCloningPolicy(
        spec["observation_dim"],
        spec["action_dim"],
        hidden=spec["network"]["hidden_sizes"][0],
    )
    policy.model.load_state_dict(
        restored_payload[spec["state_dict_key"]]
    )
    observation = np.asarray([[0.2, -0.3]], dtype=np.float32)
    assert policy.normalized_action(observation) == pytest.approx(
        trainer.policy.normalized_action(observation)
    )

    resumed = BehaviorCloningTrainer(
        DatasetReplay(dataset_path, seed=999),
        hidden=16,
        learning_rate=1e-2,
        seed=999,
    )
    resumed.load_state_dict(restored_payload["trainer"])
    assert resumed.steps == trainer.steps
    assert resumed.policy.normalized_action(
        observation
    ) == pytest.approx(trainer.policy.normalized_action(observation))


def test_bc_backend_writes_policy_spec_and_policy_only_state(
    tmp_path,
):
    dataset_path = _dataset(tmp_path, episode_count=1)
    checkpoint_path = tmp_path / "bc-backend.pt"
    config = RLTrainingConfig(
        track_id="quadruple-regulation-generalist-v1",
        algorithm_id="bc",
        training_seed=0,
        total_transitions=1,
        n_envs=1,
        algorithm={"hidden": 16},
        output={
            "directory": str(tmp_path),
            "name": "bc-backend",
        },
        dataset_id="phase-e-prior-v2",
        dataset_path=str(dataset_path),
    )

    plan = resolve_training_plan(config)
    assert plan.policy_path == checkpoint_path
    result = run_bc(plan)

    assert result.policy_path == checkpoint_path
    assert result.final_step == 1
    assert result.checkpoint_selection == "final-fixed-dataset"
    assert result.training_metadata["optimizer_steps"] == 1
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["policy_spec"]["algorithm_id"] == "bc"
    assert checkpoint["policy_spec"]["network"]["hidden_sizes"] == [
        16,
        16,
    ]
    assert set(checkpoint["policy_state_dict"]) == set(
        checkpoint["trainer"]["policy"]
    )


@pytest.mark.parametrize("strict", [False, True])
def test_optional_export_failure_preserves_native_checkpoint_and_artifact(
    tmp_path, monkeypatch, strict
):
    dataset_path = _dataset(tmp_path, episode_count=1)
    name = "bc-strict-export" if strict else "bc-best-effort-export"
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.export_policy_checkpoint",
        lambda *args, **kwargs: ExportResult(
            format="onnx",
            status="failed",
            path=str(tmp_path / f"{name}.onnx"),
            error="RuntimeError: injected export failure",
        ),
    )
    track = load_track("quadruple-regulation-generalist-v1")
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.load_policy_checkpoint",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.evaluate_validation_policy",
        lambda controller, plan, **kwargs: {
            "split": "validation",
            "track_id": track.id,
            "track_hash": track.track_hash,
            "episode_plan_hash": plan.plan_hash,
            "seed_namespace": track.seed_namespace("validation"),
            "base_seeds": list(plan.base_seeds),
            "case_count": 0,
            "results": [],
            "aggregate": {
                "metric": "regulation_cost_rate",
                "metric_direction": "minimize",
                "metric_value": 0.0,
                "official_score": 0.0,
                "ranking_eligible": True,
                "case_values": [0.0],
            },
        },
    )
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.check_benchmark_artifacts",
        lambda *args, **kwargs: {"ok": True, "failed": []},
    )
    config = RLTrainingConfig(
        track_id="quadruple-regulation-generalist-v1",
        algorithm_id="bc",
        training_seed=0,
        total_transitions=1,
        n_envs=1,
        algorithm={"batch_size": 2, "hidden": 8},
        output={
            "directory": str(tmp_path),
            "name": name,
            "onnx": True,
            "strict_export": strict,
            "save_rollout": False,
        },
        dataset_id="phase-e-prior-v2",
        dataset_path=str(dataset_path),
        validation_seeds=(7100,),
    )
    if strict:
        with pytest.raises(RuntimeError, match="preserving native checkpoint"):
            run_experiment(config)
    else:
        run_experiment(config)
    assert (tmp_path / f"{name}.pt").is_file()
    artifact = tmp_path / f"{name}_artifacts" / "benchmark.json"
    assert artifact.is_file()
    training = json.loads(artifact.read_text())["training"]
    assert training["exports"]["onnx"]["status"] == "failed"
    assert "injected export failure" in (
        training["exports"]["onnx"]["error"]
    )


def test_rlpd_actor_only_checkpoint_excludes_training_state(tmp_path):
    agent = RLPD(
        2,
        1,
        hidden=16,
        n_critics=2,
        subset=1,
        batch=4,
        scenario="quadruple",
        action_mode="actuator",
        seed=11,
    )
    checkpoint = agent.inference_checkpoint()
    path = tmp_path / "rlpd-actor.pt"
    torch.save(checkpoint, path)
    restored_payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    assert set(restored_payload) == {
        "policy_spec",
        "policy_state_dict",
    }
    assert {
        "critics",
        "targets",
        "actor_optimizer",
        "critic_optimizer",
        "alpha_optimizer",
        "online_replay",
        "offline_replay",
    }.isdisjoint(restored_payload)
    spec = validate_policy_spec(restored_payload["policy_spec"])
    actor = Actor(
        spec["observation_dim"],
        spec["action_dim"],
        hidden=spec["network"]["hidden_sizes"][0],
    )
    actor.load_state_dict(restored_payload[spec["state_dict_key"]])
    observation = torch.as_tensor(
        [[0.2, -0.3]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        mu, _ = actor(observation)
        reconstructed = torch.tanh(mu).numpy()
    expected = agent.policy_action_batch(
        observation.numpy(),
        deterministic=True,
    )
    assert reconstructed == pytest.approx(expected)


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
    assert state["policy_spec"]["algorithm_id"] == "rlpd"
    assert set(state["policy_state_dict"]) == set(
        agent.actor.state_dict()
    )
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

    incompatible = dict(state)
    incompatible["schema_version"] = "aiogym.rlpd_state.v2"
    rejected = RLPD(
        2,
        1,
        hidden=16,
        n_critics=2,
        subset=1,
        batch=4,
        online_capacity=32,
        seed=1000,
    )
    with pytest.raises(ValueError, match="unsupported RLPD checkpoint"):
        rejected.load_state_dict(incompatible)


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
        "reward_spec_id": "regulation-v1",
        "reward_spec_hash": get_reward_spec("regulation-v1").spec_hash,
        "env_spec_hash": "0" * 64,
        "env_spec_hash_schema": "aiogym.resolved_env_spec.v2",
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

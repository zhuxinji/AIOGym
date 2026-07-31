from __future__ import annotations

import hashlib

import numpy as np
import pytest

from aiogym.benchmarks import load_track
from aiogym.datasets.collector import collect_episode
from aiogym.datasets.collector_adapters import make_collector_behavior
from aiogym.datasets.writer import file_sha256
from aiogym.rl.episode_env import (
    make_track_episode_sampler,
    make_track_training_base_env,
)


TRACK_ID = "quadruple-regulation-generalist-v1"


def _collect(collector_id, *, seed=7, options=None, max_steps=24):
    track = load_track(TRACK_ID)
    sampler = make_track_episode_sampler(track)
    episode_spec = sampler.sample(seed)
    env = make_track_training_base_env(track, sampler=sampler)
    behavior = make_collector_behavior(
        collector_id,
        env,
        episode_spec,
        options=options,
    )
    try:
        return collect_episode(
            behavior.env,
            episode_spec,
            collector_id=collector_id,
            policy=behavior.policy,
            checkpoint_hash=behavior.checkpoint_hash,
            track_id=track.id,
            episode_index=0,
            max_steps=max_steps,
        )
    finally:
        behavior.env.close()


def test_pid_collector_behaviors_are_seeded_and_semantically_distinct():
    nominal = _collect("nominal_pid")
    randomized = _collect("parameter_randomized_pid")
    randomized_replay = _collect("parameter_randomized_pid")
    noisy = _collect("noisy_pid")

    def trace_hash(episode):
        return hashlib.sha256(
            episode.array("action_policy_normalized").tobytes()
        ).hexdigest()

    assert randomized.content_hash == randomized_replay.content_hash
    assert len(
        {
            trace_hash(nominal),
            trace_hash(randomized),
            trace_hash(noisy),
        }
    ) == 3
    tuning = randomized.metadata["collector_behavior"]["sampled_tuning"]
    assert tuning
    assert np.all(
        np.abs(randomized.array("action_policy_normalized")) <= 1.0
    )
    noise = noisy.metadata["collector_behavior"]
    assert noise["noise"]["kind"] == "low_pass_gaussian"
    assert "action_clip_statistics" in noise


@pytest.mark.parametrize(
    "collector_id",
    ("nominal_pid", "noisy_pid", "safe_excitation"),
)
def test_collectors_record_exact_reset_state_and_requested_diagnostics(
    collector_id,
):
    episode = _collect(collector_id, max_steps=4)
    metadata = episode.metadata
    true_state = episode.array("true_state")
    recorded = np.asarray(
        metadata["initial_state"],
        dtype=true_state.dtype,
    )
    requested = np.asarray(
        metadata["requested_initial_state"],
        dtype=np.float64,
    )
    assert np.array_equal(recorded, true_state[0])
    assert metadata["reset_state_delta_linf"] == float(
        np.max(
            np.abs(
                requested - recorded.astype(np.float64)
            )
        )
    )


def test_safe_excitation_is_projected_through_declared_shield():
    episode = _collect(
        "safe_excitation",
        options={"max_delta_per_step": 0.03},
    )
    shield = episode.metadata["safety_shield"]
    assert shield["shield_id"] == (
        "dataset-safe-excitation-projection-v1"
    )
    assert shield["max_delta_per_step"] == 0.03
    action = episode.array("action_commanded_physical")
    maximum_voltage = episode.metadata["plant_parameters"]["max_voltage"]
    normalized_command = action / float(maximum_voltage)
    assert (
        np.max(np.abs(np.diff(normalized_command, axis=0)))
        <= 0.030001
    )


def test_checkpoint_collector_verifies_and_records_sha256(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"versioned-checkpoint")
    digest = file_sha256(checkpoint)

    class FakePolicy:
        name = "fake-checkpoint"
        action_mode = "actuator"
        control_structure = "fake"
        controller_api_version = "aiogym.controller.v1"

        def reset(self, seed=None):
            del seed

        def act(self, observation, context):
            del observation, context
            return np.full(2, 0.5, dtype=np.float32)

        def metadata(self):
            return {
                "name": self.name,
                "kind": "fake",
                "checkpoint": dict(self.checkpoint_metadata),
            }

    def load_policy(spec):
        policy = FakePolicy()
        policy.checkpoint_metadata = {
            "path": str(spec.path),
            "algorithm_id": spec.algorithm_id,
            "sha256": spec.sha256,
        }
        return policy

    monkeypatch.setattr(
        "aiogym.datasets.collector_adapters.load_policy_checkpoint",
        load_policy,
    )
    episode = _collect(
        "checkpoint",
        options={
            "checkpoints": [
                {
                    "path": str(checkpoint),
                    "algorithm_id": "sac",
                    "sha256": digest,
                    "weight": 1.0,
                }
            ]
        },
        max_steps=4,
    )
    assert episode.metadata["checkpoint_hash"] == digest
    assert (
        episode.metadata["collector_behavior"]["checkpoint"]["sha256"]
        == digest
    )


def test_bc_checkpoint_collector_scales_native_action_once(tmp_path):
    torch = pytest.importorskip("torch")
    from aiogym.rl.behavior_cloning import BehaviorCloningPolicy
    from aiogym.rl.policy_spec import behavior_cloning_policy_spec

    track = load_track(TRACK_ID)
    sampler = make_track_episode_sampler(track)
    env = make_track_training_base_env(track, sampler=sampler)
    try:
        observation_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
    finally:
        env.close()
    policy = BehaviorCloningPolicy(
        observation_dim,
        action_dim,
        hidden=4,
    )
    with torch.no_grad():
        for parameter in policy.model.parameters():
            parameter.zero_()
        policy.model[-2].bias.fill_(float(np.arctanh(0.5)))
    checkpoint = tmp_path / "bc.pt"
    torch.save(
        {
            "policy_spec": behavior_cloning_policy_spec(
                observation_dim,
                action_dim,
                4,
                scenario="quadruple",
                action_mode="actuator",
            ),
            "policy_state_dict": policy.model.state_dict(),
        },
        checkpoint,
    )
    digest = file_sha256(checkpoint)

    episode = _collect(
        "checkpoint",
        options={
            "checkpoints": [
                {
                    "path": str(checkpoint),
                    "algorithm_id": "bc",
                    "sha256": digest,
                }
            ]
        },
        max_steps=3,
    )

    assert episode.array("action_policy_normalized") == pytest.approx(
        0.5
    )
    assert episode.metadata["checkpoint_hash"] == digest


def test_recovery_collector_requires_and_records_recovery_episode():
    track = load_track("cascade-recovery-diagnostic-v1")
    sampler = make_track_episode_sampler(track)
    episode_spec = sampler.sample(5)
    env = make_track_training_base_env(track, sampler=sampler)
    behavior = make_collector_behavior(
        "recovery",
        env,
        episode_spec,
    )
    try:
        episode = collect_episode(
            behavior.env,
            episode_spec,
            collector_id="recovery",
            policy=behavior.policy,
            track_id=track.id,
            episode_index=0,
            max_steps=8,
        )
    finally:
        behavior.env.close()
    assert (
        episode.metadata["collector_behavior"]["control_structure"]
        == "boundary_recovery_pid"
    )
    assert "initial_safety_debt" in episode.metadata["recovery_audit"]
    assert "controller_created_violation" in (
        episode.metadata["recovery_audit"]
    )

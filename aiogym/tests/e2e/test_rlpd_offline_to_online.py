from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.e2e
pytest.importorskip("torch")

from aiogym import load_track
from aiogym.datasets.schema import DatasetEpisode
from aiogym.datasets.writer import DatasetWriter
from aiogym.rl.rlpd import RLPD
from aiogym.rl.validation import CompleteValidationCallback
from aiogym.rewards import get_reward_spec


TRACK_ID = "quadruple-regulation-generalist-v1"


def test_rlpd_hybrid_accounting_and_validation_only_selection(tmp_path):
    dataset_path = tmp_path / "dataset"
    with DatasetWriter(
        dataset_path,
        dataset_id="rlpd-e2e-v2",
        split="training",
    ) as writer:
        writer.append_episode(_episode())

    agent = RLPD(
        2,
        1,
        hidden=8,
        n_critics=2,
        subset=1,
        utd=2,
        batch=2,
        online_capacity=8,
        offline_fraction=0.5,
        seed=4,
    )
    offline = agent.load_dataset(dataset_path, verify_checksums=True)
    sample = offline.sample(1)
    agent.push(
        sample["observation"][0],
        sample["action_policy_normalized"][0],
        sample["reward_scalar"][0],
        sample["next_observation"][0],
        sample["terminated"][0],
        sample["truncated"][0],
        sample["bootstrap_mask"][0],
    )
    agent.update()

    accounting = agent.accounting()
    assert accounting["offline_transitions"] == 8
    assert accounting["online_transitions"] == 1
    assert accounting["gradient_updates"] == 2
    assert accounting["offline_samples"] == accounting["online_samples"]
    assert accounting["sampled_offline_fraction"] == 0.5

    visited = []
    track = load_track(TRACK_ID)

    def evaluate(controller, resolved_track, **kwargs):
        assert controller is agent
        assert resolved_track.track_hash == track.track_hash
        visited.append("split" not in kwargs)
        return {
            "track_id": track.id,
            "track_hash": track.track_hash,
            "split": "validation",
            "results": [
                {
                    "case_id": case.case_id,
                    "ranking_eligible": True,
                    "safety_gate": {"reasons": []},
                }
                for case in track.resolved_cases("validation")
            ],
            "aggregate": {
                "metric": "regulation_cost_rate",
                "metric_direction": "minimize",
                "metric_value": 1.0,
                "case_values": [1.0, 1.0],
                "case_count": 2,
                "ranking_eligible": True,
            },
        }

    callback = CompleteValidationCallback(
        track,
        base_seeds=[7101],
        evaluate_fn=evaluate,
    )
    callback.evaluate(agent, checkpoint_id="rlpd-step-1", step=1)
    assert visited == [True]
    assert callback.selector.best.checkpoint_id == "rlpd-step-1"


def _episode() -> DatasetEpisode:
    count = 8
    observation = np.linspace(
        -0.5,
        0.5,
        count * 2,
        dtype=np.float32,
    ).reshape(count, 2)
    action = np.tanh(observation[:, :1]).astype(np.float32)
    terminated = np.zeros(count, dtype=np.bool_)
    truncated = np.zeros(count, dtype=np.bool_)
    truncated[-1] = True
    zeros = np.zeros(count, dtype=np.float32)
    return DatasetEpisode(
        metadata={
            "episode_id": "rlpd-e2e-0",
            "split": "training",
            "track_id": "rlpd-e2e-track",
            "distribution_id": "rlpd-e2e-distribution",
            "distribution_hash": "distribution-hash",
            "episode_spec_id": "rlpd-e2e-spec",
            "resolved_hash": "resolved-hash",
            "base_seed": 4,
            "component_seeds": {"policy": 4},
            "scenario": "quadruple",
            "goal": "regulation",
            "reward_spec_id": "regulation-v1",
            "reward_spec_hash": get_reward_spec("regulation-v1").spec_hash,
            "env_spec_hash": "0" * 64,
            "env_spec_hash_schema": "aiogym.resolved_env_spec.v2",
            "action_mode": "actuator",
            "collector_id": "nominal_pid",
            "policy_id": "pid",
            "collector_quality_tag": "expert",
            "plant_parameters": {},
            "initial_state": observation[0].tolist(),
            "reference_schedule": [{"at_step": 0, "values": [0.0]}],
            "disturbance_schedule": [],
            "sensor_model": {"kind": "none"},
            "actuator_model": {"kind": "none"},
            "termination_reason": "time_limit",
            "difficulty_tags": ["L0"],
            "summary": {},
        },
        observation=observation,
        true_state=observation,
        reference=np.zeros((count, 1), dtype=np.float32),
        measured_disturbance=np.empty((count, 0), dtype=np.float32),
        action_policy_normalized=action,
        action_commanded_physical=0.5 * (action + 1.0),
        action_applied_physical=0.5 * (action + 1.0),
        reward_scalar=zeros,
        reward_terms={"tracking": zeros},
        cost_channels={"safety": zeros},
        step_index=np.arange(count, dtype=np.int64),
        physical_time=np.arange(count, dtype=np.float64),
        next_observation=observation + 0.01,
        next_true_state=observation + 0.01,
        terminated=terminated,
        truncated=truncated,
        bootstrap_mask=np.ones(count, dtype=np.float32),
    )

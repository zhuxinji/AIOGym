"""Migration from legacy in-memory TransitionDataset rows to Dataset v2."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from aiogym.compat.transitions import TransitionDataset

from .schema import DatasetEpisode


def migrate_transition_dataset(
    dataset: TransitionDataset,
    *,
    dataset_id: str,
    track_id: str,
    scenario: str,
    goal: str = "regulation",
    split: str = "training",
    collector_id: str = "legacy-v1",
    policy_id: str = "legacy-policy",
    collector_quality_tag: str = "legacy",
    control_dt: float = 1.0,
) -> tuple[DatasetEpisode, ...]:
    """Convert each legacy episode atomically without mixing boundaries."""

    if not isinstance(dataset, TransitionDataset):
        raise TypeError("dataset must be a TransitionDataset")
    grouped = defaultdict(list)
    for transition in dataset:
        grouped[transition.episode].append(transition)
    episodes = []
    for episode_index in sorted(grouped):
        transitions = sorted(
            grouped[episode_index],
            key=lambda item: item.step,
        )
        episodes.append(
            _migrate_episode(
                transitions,
                dataset_id=dataset_id,
                track_id=track_id,
                scenario=scenario,
                goal=goal,
                split=split,
                collector_id=collector_id,
                policy_id=policy_id,
                collector_quality_tag=collector_quality_tag,
                control_dt=control_dt,
                episode_index=episode_index,
            )
        )
    return tuple(episodes)


def _migrate_episode(
    transitions,
    *,
    dataset_id,
    track_id,
    scenario,
    goal,
    split,
    collector_id,
    policy_id,
    collector_quality_tag,
    control_dt,
    episode_index,
):
    if not transitions:
        raise ValueError("cannot migrate an empty episode")
    action = np.asarray(
        [item.action for item in transitions],
        dtype=np.float32,
    )
    reward_rows = [
        dict(item.info.get("reward_terms") or {}) for item in transitions
    ]
    cost_rows = [dict(item.info.get("costs") or {}) for item in transitions]
    reward_terms = _channel_rows(reward_rows)
    cost_channels = _channel_rows(cost_rows)
    reference_schedule = _changed_schedule(
        [item.setpoint for item in transitions],
        field="values",
    )
    disturbance_schedule = _disturbance_schedule(transitions)
    final = transitions[-1]
    termination_reason = str(
        final.info.get("termination_reason")
        or (
            "time_limit"
            if final.truncated
            else "terminal"
            if final.terminated
            else "legacy_incomplete"
        )
    )
    resolved_hash = str(
        final.info.get("episode_spec_hash")
        or f"legacy-v1-resolved-{episode_index}"
    )
    distribution_hash = str(
        final.info.get("distribution_hash")
        or "legacy-v1-distribution"
    )
    episode_spec_id = str(
        final.info.get("episode_spec_id")
        or f"legacy-v1-episode-spec-{episode_index}"
    )
    metadata = {
        "episode_id": f"{dataset_id}:legacy-episode-{episode_index}",
        "split": split,
        "track_id": track_id,
        "distribution_id": str(
            final.info.get("distribution_id") or "legacy-v1"
        ),
        "distribution_hash": distribution_hash,
        "episode_spec_id": episode_spec_id,
        "resolved_hash": resolved_hash,
        "base_seed": int(final.info.get("episode_base_seed", 0)),
        "component_seeds": dict(
            final.info.get("episode_component_seeds") or {}
        ),
        "scenario": scenario,
        "goal": goal,
        "action_mode": "actuator",
        "collector_id": collector_id,
        "policy_id": policy_id,
        "collector_quality_tag": collector_quality_tag,
        "plant_parameters": dict(final.info.get("plant_parameters") or {}),
        "initial_state": list(transitions[0].state),
        "reference_schedule": reference_schedule,
        "disturbance_schedule": disturbance_schedule,
        "sensor_model": {"kind": "legacy_unspecified"},
        "actuator_model": {"kind": "legacy_identity_assumed"},
        "difficulty_tags": ["legacy-v1"],
        "termination_reason": termination_reason,
        "summary": {
            "transitions": len(transitions),
            "return": float(sum(item.reward for item in transitions)),
            "cost_totals": {
                name: float(np.sum(values))
                for name, values in cost_channels.items()
            },
            "terminated": bool(final.terminated),
            "truncated": bool(final.truncated),
        },
        "migration": {
            "source_schema": "aiogym.transition.v1",
            "legacy_action_alias": "action_commanded_physical",
        },
    }
    return DatasetEpisode(
        metadata=metadata,
        observation=[item.obs for item in transitions],
        true_state=[item.state for item in transitions],
        reference=[item.setpoint for item in transitions],
        measured_disturbance=None,
        action_policy_normalized=2.0 * action - 1.0,
        action_commanded_physical=action,
        action_applied_physical=action,
        reward_scalar=[item.reward for item in transitions],
        reward_terms=reward_terms,
        cost_channels=cost_channels,
        next_observation=[item.next_obs for item in transitions],
        next_true_state=[item.next_state for item in transitions],
        terminated=[item.terminated for item in transitions],
        truncated=[item.truncated for item in transitions],
        bootstrap_mask=[
            0.0 if item.terminated else 1.0 for item in transitions
        ],
        step_index=list(range(len(transitions))),
        physical_time=[
            (index + 1) * float(control_dt)
            for index in range(len(transitions))
        ],
    )


def _changed_schedule(values, *, field):
    schedule = []
    previous = None
    for index, value in enumerate(values):
        current = list(value)
        if previous is None or current != previous:
            schedule.append({"at_step": index, field: current})
            previous = current
    return schedule


def _disturbance_schedule(transitions):
    events = []
    previous = {}
    for transition in transitions:
        current = dict(transition.disturbance)
        for name, value in current.items():
            if name not in previous or previous[name] != value:
                events.append(
                    {
                        "at_step": transition.step,
                        "name": str(name),
                        "value": value,
                    }
                )
        previous = current
    return events


def _channel_rows(rows):
    names = sorted({name for row in rows for name in row})
    return {
        name: np.asarray(
            [float(row.get(name, 0.0)) for row in rows],
            dtype=np.float32,
        )
        for name in names
    }


__all__ = ["migrate_transition_dataset"]

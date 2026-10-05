"""Collect policy transitions from an already constructed environment."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aiogym.controllers.base import resolve_policy
from aiogym.core.rollout import rollout

from ._metadata import environment_metadata, policy_metadata_for_environment
from .dataset import DatasetReader, DatasetWriter


def collect(
    *,
    env,
    output: str | Path,
    policy="random",
    episodes: int = 1,
    seed: int = 0,
    max_steps: int | None = None,
):
    """Collect episodes using a controller name, Policy, or ``act(observation)``.

    Use ``FunctionPolicy`` when inference needs an episode reset or context.
    The caller retains ownership of ``env``.
    """

    count = _positive_int("episodes", episodes)
    base_seed = _nonnegative_int("seed", seed)
    resolved_policy = resolve_policy(policy, env=env)
    resolved_policy_metadata = policy_metadata_for_environment(resolved_policy, env)
    writer = DatasetWriter(
        output,
        environment=environment_metadata(env),
        policy=resolved_policy_metadata,
        base_seed=base_seed,
    )
    for index in range(count):
        episode_seed = base_seed + index
        result = rollout(
            env,
            resolved_policy,
            seed=episode_seed,
            max_steps=max_steps,
            policy_metadata=resolved_policy_metadata,
        )
        arrays, metadata = _episode_arrays(result)
        writer.append(index, episode_seed, arrays, metadata=metadata)
    reader = DatasetReader(output)
    return {
        "schema_version": "aiogym.collect.v2",
        "path": str(Path(output).resolve()),
        "episodes": len(reader),
        "transitions": reader.transition_count,
        "metadata": reader.metadata,
    }


def _episode_arrays(result):
    transitions = result.transitions
    reward_names = sorted(
        {name for row in transitions for name in row.info["reward_terms"]}
    )
    constraint_names = sorted(
        {name for row in transitions for name in row.info["constraint_costs"]}
    )
    safety_margin_names = sorted(
        {name for row in transitions for name in row.info["safety_margins"]}
    )
    disturbance_names = sorted(
        {
            name
            for row in transitions
            for field in ("disturbance", "transition_disturbance")
            for name in row.info[field]
        }
    )

    def stack_info(name):
        return np.asarray([row.info[name] for row in transitions])

    arrays = {
        "observation": np.asarray(
            [row.observation for row in transitions], dtype=np.float32
        ),
        "action": np.asarray([row.action for row in transitions], dtype=np.float32),
        "reward": np.asarray([row.reward for row in transitions], dtype=np.float64),
        "next_observation": np.asarray(
            [row.next_observation for row in transitions], dtype=np.float32
        ),
        "terminated": np.asarray(
            [row.terminated for row in transitions], dtype=np.bool_
        ),
        "truncated": np.asarray(
            [row.truncated for row in transitions], dtype=np.bool_
        ),
        "step_index": np.arange(len(transitions), dtype=np.int64),
        "physical_time": np.asarray(
            [row.physical_time for row in transitions], dtype=np.float64
        ),
        "true_state": stack_info("true_state").astype(np.float32),
        "reference": stack_info("reference").astype(np.float32),
        "transition_reference": stack_info("transition_reference").astype(
            np.float32
        ),
        "commanded_action": stack_info("commanded_action").astype(np.float32),
        "channel_action": stack_info("channel_action").astype(np.float32),
        "applied_action": stack_info("applied_action").astype(np.float32),
        "minimum_safety_margin": stack_info("minimum_safety_margin").astype(
            np.float64
        ),
        "reward_terms": np.asarray(
            [
                [float(row.info["reward_terms"][name]) for name in reward_names]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(reward_names)),
        "constraint_costs": np.asarray(
            [
                [float(row.info["constraint_costs"][name]) for name in constraint_names]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(constraint_names)),
        "safety_margins": np.asarray(
            [
                [float(row.info["safety_margins"][name]) for name in safety_margin_names]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(safety_margin_names)),
        "disturbance": np.asarray(
            [
                [float(row.info["disturbance"][name]) for name in disturbance_names]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(disturbance_names)),
        "transition_disturbance": np.asarray(
            [
                [
                    float(row.info["transition_disturbance"][name])
                    for name in disturbance_names
                ]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(disturbance_names)),
    }
    metadata = {
        "reward_term_names": reward_names,
        "constraint_cost_names": constraint_names,
        "safety_margin_names": safety_margin_names,
        "disturbance_names": disturbance_names,
        "episode_spec": dict(result.reset_info["episode_spec"]),
        "episode_family": result.reset_info["episode_family"],
        "episode_parameters": dict(result.reset_info["episode_parameters"]),
        "runtime_variation": dict(result.reset_info["runtime_variation"]),
    }
    return arrays, metadata


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = ["collect"]

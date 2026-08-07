"""Policy-agnostic Dataset v3 collection through core.rollout."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import aiogym.scenarios  # noqa: F401
from aiogym.core import (
    resolve_condition_alias,
    rollout,
)

from ._policy_target import resolve_policy_target
from .dataset import DatasetReader, DatasetWriter


def collect(
    *,
    task: str,
    output: str | Path,
    plant=None,
    condition=None,
    preset: str | None = None,
    policy="random",
    episodes: int = 1,
    seed: int = 0,
    dataset_id: str | None = None,
    resume: bool = False,
    max_steps: int | None = None,
    allow_plant_transfer: bool = False,
    allow_condition_transfer: bool = False,
):
    count = _positive_int("episodes", episodes)
    base_seed = _nonnegative_int("seed", seed)
    condition = resolve_condition_alias(condition, preset)
    target = resolve_policy_target(
        policy,
        task=task,
        plant=plant,
        condition=condition,
        allow_plant_transfer=allow_plant_transfer,
        allow_condition_transfer=allow_condition_transfer,
    )
    env = target.env
    resolved_policy = target.policy
    identifier = dataset_id or Path(output).name
    try:
        writer = DatasetWriter(
            output,
            dataset_id=identifier,
            task_id=env.task.id,
            task_hash=env.task.task_hash,
            plant_id=env.plant.id,
            plant_hash=env.plant.plant_hash,
            condition_id=env.condition.id,
            condition_hash=env.condition.condition_hash,
            interface_hash=env.identity.interface_hash,
            env_hash=env.identity.env_hash,
            policy=resolved_policy.metadata(),
            policy_training_contract=target.training_contract,
            target_environment_contract=target.target_contract,
            contract_status=target.contract_status,
            transfer_flags=target.transfer_flags,
            base_seed=base_seed,
            state_schema={"fields": list(env.model.state_schema())},
            observation_schema=_space_schema(env.observation_space),
            action_schema={
                "fields": list(env.model.action_schema()),
                "space": _space_schema(env.action_space),
            },
            resume=resume,
        )
        for index in range(len(writer.manifest["episodes"]), count):
            episode_seed = base_seed + index
            result = rollout(
                env,
                resolved_policy,
                seed=episode_seed,
                max_steps=max_steps,
            )
            arrays, metadata = _episode_arrays(result)
            writer.append(index, episode_seed, arrays, metadata=metadata)
    finally:
        if target.created_env:
            env.close()
    reader = DatasetReader(output, verify_checksums=True)
    return {
        "schema_version": "aiogym.collect_result.v1",
        "dataset_id": identifier,
        "path": str(Path(output).resolve()),
        "task_id": env.task.id,
        "task_hash": env.task.task_hash,
        "plant_id": env.plant.id,
        "plant_hash": env.plant.plant_hash,
        "condition_id": env.condition.id,
        "condition_hash": env.condition.condition_hash,
        "interface_hash": env.identity.interface_hash,
        "env_hash": env.identity.env_hash,
        "policy_training_contract": target.training_contract,
        "target_environment_contract": dict(target.target_contract),
        "contract_status": target.contract_status,
        "transfer_flags": dict(target.transfer_flags),
        "episodes": len(reader),
        "transitions": reader.transition_count,
        "manifest": reader.manifest,
    }


def _episode_arrays(result):
    transitions = result.transitions
    reward_names = sorted(
        {name for row in transitions for name in row.info.get("reward_terms", {})}
    )
    constraint_names = sorted(
        {name for row in transitions for name in row.info.get("constraint_costs", {})}
    )
    disturbance_names = sorted(
        {name for row in transitions for name in row.info.get("disturbance", {})}
    )

    def stack_info(name, fallback):
        return np.asarray([row.info.get(name, fallback(row)) for row in transitions])

    arrays = {
        "observation": np.asarray([row.observation for row in transitions], dtype=np.float32),
        "action": np.asarray([row.action for row in transitions], dtype=np.float32),
        "reward": np.asarray([row.reward for row in transitions], dtype=np.float64),
        "next_observation": np.asarray(
            [row.next_observation for row in transitions], dtype=np.float32
        ),
        "terminated": np.asarray([row.terminated for row in transitions], dtype=np.bool_),
        "truncated": np.asarray([row.truncated for row in transitions], dtype=np.bool_),
        "step_index": np.arange(len(transitions), dtype=np.int64),
        "physical_time": np.asarray(
            [row.physical_time for row in transitions], dtype=np.float64
        ),
        "true_state": stack_info("true_state", lambda row: row.next_observation).astype(
            np.float32
        ),
        "reference": stack_info("reference", lambda row: np.empty(0)).astype(np.float32),
        "commanded_action": stack_info("commanded_action", lambda row: row.action).astype(
            np.float32
        ),
        "applied_action": stack_info("applied_action", lambda row: row.action).astype(
            np.float32
        ),
        "reward_terms": np.asarray(
            [
                [float(row.info.get("reward_terms", {}).get(name, 0.0)) for name in reward_names]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(reward_names)),
        "constraint_costs": np.asarray(
            [
                [
                    float(row.info.get("constraint_costs", {}).get(name, 0.0))
                    for name in constraint_names
                ]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(constraint_names)),
        "disturbance": np.asarray(
            [
                [float(row.info.get("disturbance", {}).get(name, 0.0)) for name in disturbance_names]
                for row in transitions
            ],
            dtype=np.float64,
        ).reshape(len(transitions), len(disturbance_names)),
    }
    metadata = {
        "reward_term_names": reward_names,
        "constraint_cost_names": constraint_names,
        "disturbance_names": disturbance_names,
        "policy": dict(result.policy_metadata),
    }
    return arrays, metadata


def _space_schema(space):
    def bound(value):
        number = float(value)
        return number if np.isfinite(number) else None

    return {
        "shape": list(space.shape),
        "dtype": str(space.dtype),
        "low": [bound(value) for value in np.asarray(space.low).reshape(-1)],
        "high": [bound(value) for value in np.asarray(space.high).reshape(-1)],
    }


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = ["collect"]

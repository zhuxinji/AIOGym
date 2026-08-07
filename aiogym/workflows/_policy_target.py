"""Single policy-to-environment contract resolver for workflows."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aiogym.controllers.base import make_controller
from aiogym.core import make_env, resolve_legacy_request


@dataclass(frozen=True)
class ResolvedPolicyTarget:
    env: Any
    policy: Any
    created_env: bool
    target_contract: Mapping[str, Any]
    training_contract: Mapping[str, Any] | None
    transfer_flags: Mapping[str, bool]
    contract_status: str


def resolve_policy_target(
    policy,
    *,
    task=None,
    plant=None,
    condition=None,
    allow_plant_transfer=False,
    allow_condition_transfer=False,
) -> ResolvedPolicyTarget:
    bound_env = getattr(policy, "env", None) if not isinstance(policy, str) else None
    explicit_selector = task is not None or plant is not None or condition is not None
    if task is None:
        if bound_env is None:
            raise ValueError("task is required for an unbound policy")
        task = bound_env.task.id
    task, plant, condition = resolve_legacy_request(task, plant, condition)

    created_env = bound_env is None
    if bound_env is None:
        env = make_env(task, plant=plant, condition=condition)
    else:
        env = bound_env
        if explicit_selector:
            comparison = make_env(
                task,
                plant=bound_env.plant.config if plant is None else plant,
                condition=bound_env.condition if condition is None else condition,
            )
            try:
                _validate_bound_contract(
                    bound_env.identity.as_dict(), comparison.identity.as_dict()
                )
            finally:
                comparison.close()

    try:
        resolved_policy = (
            make_controller(policy, env=env) if isinstance(policy, str) else policy
        )
        metadata = dict(resolved_policy.metadata())
        training_contract = metadata.get("training_contract")
        target_contract = env.identity.as_dict()
        transfer_flags = {
            "is_transfer": False,
            "plant_changed": False,
            "condition_changed": False,
        }
        if training_contract is not None:
            training_contract = dict(training_contract)
            transfer_flags = _validate_training_contract(
                training_contract,
                target_contract,
                allow_plant_transfer=bool(allow_plant_transfer),
                allow_condition_transfer=bool(allow_condition_transfer),
            )
            contract_status = (
                "transfer" if transfer_flags["is_transfer"] else "verified"
            )
        elif isinstance(policy, str) or bound_env is not None:
            contract_status = "bound"
        else:
            contract_status = "unverified"
        return ResolvedPolicyTarget(
            env=env,
            policy=resolved_policy,
            created_env=created_env,
            target_contract=target_contract,
            training_contract=training_contract,
            transfer_flags=transfer_flags,
            contract_status=contract_status,
        )
    except Exception:
        if created_env:
            env.close()
        raise


def _validate_bound_contract(bound, requested):
    labels = {
        "task_hash": "task",
        "plant_hash": "plant",
        "condition_hash": "condition",
        "interface_hash": "interface",
        "env_hash": "environment",
    }
    for name, label in labels.items():
        if bound.get(name) != requested.get(name):
            raise ValueError(
                f"bound policy {name} does not match requested {label}"
            )


def _validate_training_contract(
    training,
    target,
    *,
    allow_plant_transfer,
    allow_condition_transfer,
):
    if training.get("task_hash") != target["task_hash"]:
        raise ValueError("checkpoint task_hash does not match target task")
    if training.get("interface_hash") != target["interface_hash"]:
        raise ValueError("checkpoint interface_hash does not match target environment")
    plant_changed = training.get("plant_hash") != target["plant_hash"]
    condition_changed = training.get("condition_hash") != target["condition_hash"]
    if plant_changed and not allow_plant_transfer:
        raise ValueError(
            "checkpoint plant_hash does not match target; "
            "set allow_plant_transfer=True to permit this transfer"
        )
    if condition_changed and not allow_condition_transfer:
        raise ValueError(
            "checkpoint condition_hash does not match target; "
            "set allow_condition_transfer=True to permit this transfer"
        )
    is_transfer = plant_changed or condition_changed
    if not is_transfer and training.get("env_hash") != target["env_hash"]:
        raise ValueError("checkpoint env_hash does not match target environment")
    return {
        "is_transfer": is_transfer,
        "plant_changed": plant_changed,
        "condition_changed": condition_changed,
    }


__all__ = ["ResolvedPolicyTarget", "resolve_policy_target"]

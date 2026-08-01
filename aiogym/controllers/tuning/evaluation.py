"""Canonical specialist evaluation helper for controller tuning."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from aiogym.controllers.registry import make_controller
from aiogym import make_env
from aiogym.evaluation.execution import evaluate_controller
from aiogym._internal.validation import seed_sequence


def evaluate_specialist(
    *,
    scenario: str,
    case: str,
    controller: str,
    seeds: Sequence[int],
    controller_config: Mapping | None = None,
    environment: Mapping | None = None,
) -> dict:
    seed_list = seed_sequence("tuning evaluation seeds", seeds)
    agent = make_controller(
        controller,
        scenario=scenario,
        config=dict(controller_config or {}),
    )
    env = make_env(
        config={
            "scenario": scenario,
            "case": case,
            "reward_spec": "regulation-v1",
            "environment": {
                "action_mode": "actuator",
                **dict(environment or {}),
            },
        }
    )
    try:
        return evaluate_controller(
            agent,
            env,
            episodes=len(seed_list),
            seed=seed_list[0],
            seed_list=seed_list,
            goal_specification="regulation",
        )
    finally:
        env.close()


__all__ = ["evaluate_specialist"]

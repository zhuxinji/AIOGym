"""Canonical specialist evaluation helper for controller tuning."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from aiogym.controllers import make_controller
from aiogym.env import AIOGymEnv
from aiogym.evaluation import evaluate_controller


def evaluate_specialist(
    *,
    scenario: str,
    case: str,
    controller: str,
    seeds: Sequence[int],
    controller_config: Mapping | None = None,
    environment: Mapping | None = None,
) -> dict:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("tuning evaluation requires at least one seed")
    agent = make_controller(
        controller,
        scenario=scenario,
        config=dict(controller_config or {}),
    )
    env = AIOGymEnv(
        scenario,
        case=case,
        reward_spec="regulation-v1",
        action_mode="actuator",
        **dict(environment or {}),
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

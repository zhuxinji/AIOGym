"""Registry for canonical versioned reward specifications."""
from __future__ import annotations

from collections.abc import Iterable

from .specs import RewardSpec


_BUILTIN_SPECS = {
    "regulation-v1": RewardSpec(
        id="regulation-v1",
        version="1",
        goal="regulation",
        term_weights={
            "tracking_error": 1.0,
            "slew": 0.0,
            "effort": 0.0,
        },
        cost_weights={
            "service_shortfall": 0.0,
            "soft_safety": 0.0,
            "hard_safety": 0.0,
            "protection_intervention": 0.0,
        },
        terminal_failure_cost_rate=1.0,
        metadata={
            "canonical": True,
            "official": True,
            "output_weights": 1.0,
            "calibration_status": "pending-track-calibration",
            "discretization": "physical-time-integral",
        },
    ),
    "economic-v1": RewardSpec(
        id="economic-v1",
        version="1",
        goal="economic",
        term_weights={
            "product_value": 1.0,
            "energy_cost": 1.0,
            "material_cost": 1.0,
            "waste_cost": 1.0,
        },
        cost_weights={
            "service_shortfall": 1.0,
            "soft_safety": 1.0,
            "hard_safety": 0.0,
            "protection_intervention": 0.0,
        },
        terminal_failure_cost_rate=1.0,
        metadata={
            "canonical": True,
            "official": True,
            "cost_weight_sources": {
                "service_shortfall": "model.economic_config.w_product_shortfall",
                "soft_safety": "model.economic_config.w_viol",
            },
            "discretization": "physical-time-integral",
        },
    ),
}


def list_reward_specs() -> tuple[str, ...]:
    return tuple(sorted(_BUILTIN_SPECS))


def get_reward_spec(spec_id: str) -> RewardSpec:
    try:
        return _BUILTIN_SPECS[str(spec_id)]
    except KeyError as exc:
        choices = ", ".join(list_reward_specs())
        raise ValueError(f"reward_spec must be one of: {choices}") from exc


def resolve_reward_spec(
    reward_spec: str | RewardSpec | None = None,
) -> RewardSpec:
    if reward_spec is None:
        return get_reward_spec("regulation-v1")
    resolved = (
        get_reward_spec(reward_spec)
        if isinstance(reward_spec, str)
        else reward_spec
    )
    if not isinstance(resolved, RewardSpec):
        raise TypeError("reward_spec must be a versioned ID or RewardSpec")
    if resolved.id not in _BUILTIN_SPECS:
        raise ValueError(
            "custom RewardSpec objects are not registered; "
            "use regulation-v1 or economic-v1"
        )
    return resolved


def iter_reward_specs() -> Iterable[RewardSpec]:
    return (get_reward_spec(spec_id) for spec_id in list_reward_specs())

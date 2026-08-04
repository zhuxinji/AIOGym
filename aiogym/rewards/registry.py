"""Registry for canonical versioned reward specifications."""
from __future__ import annotations

from collections.abc import Iterable

from .specs import RewardSpec


REWARD_SELECTORS = {
    "regulation": "regulation-v1",
    "economic": "economic-v1",
}

_REWARD_DISPLAY = {
    "regulation-v1": {
        "selector": "regulation",
        "title": "Set-point regulation",
        "summary": "Set-point tracking; safety is reported separately.",
        "term_descriptions": {
            "tracking_error": "Physical-time integral of tracking error.",
            "slew": "Actuator movement penalty.",
            "effort": "Actuator effort penalty.",
        },
        "cost_descriptions": {
            "service_shortfall": "Service delivery shortfall.",
            "soft_safety": "Soft safety-envelope violation.",
            "hard_safety": "Hard safety violation.",
            "protection_intervention": "Protective action intervention.",
        },
    },
    "economic-v1": {
        "selector": "economic",
        "title": "Economic operation",
        "summary": (
            "Product value minus energy, material, waste, and service costs."
        ),
        "term_descriptions": {
            "product_value": "Value of useful production.",
            "energy_cost": "Energy consumption cost.",
            "material_cost": "Material consumption cost.",
            "waste_cost": "Waste handling or disposal cost.",
        },
        "cost_descriptions": {
            "service_shortfall": "Service delivery shortfall.",
            "soft_safety": "Soft safety-envelope violation.",
            "hard_safety": "Hard safety violation.",
            "protection_intervention": "Protective action intervention.",
        },
    },
}


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


def resolve_reward_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("reward selector must be a non-empty string")
    resolved = REWARD_SELECTORS.get(value, value)
    if resolved not in _BUILTIN_SPECS:
        selectors = ", ".join(REWARD_SELECTORS)
        canonical = ", ".join(list_reward_specs())
        raise ValueError(
            f"unknown reward selector or ID {value!r}; selectors: "
            f"{selectors}; canonical IDs: {canonical}"
        )
    return resolved


def reward_display(spec_id: str) -> dict:
    canonical_id = resolve_reward_id(spec_id)
    return {
        "reward_spec_id": canonical_id,
        **{
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in _REWARD_DISPLAY[canonical_id].items()
        },
    }


def iter_reward_catalog() -> tuple[dict, ...]:
    return tuple(
        {
            **reward_display(selector),
            "goal": get_reward_spec(spec_id).goal,
        }
        for selector, spec_id in REWARD_SELECTORS.items()
    )


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
        get_reward_spec(resolve_reward_id(reward_spec))
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
    canonical = get_reward_spec(resolved.id)
    if resolved.as_dict() != canonical.as_dict():
        raise ValueError(
            "RewardSpec content does not match the registered canonical "
            f"definition for {resolved.id!r}"
        )
    return canonical


def iter_reward_specs() -> Iterable[RewardSpec]:
    return (get_reward_spec(spec_id) for spec_id in list_reward_specs())

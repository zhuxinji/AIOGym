"""Compile fixed Case profiles into programmatic distributions."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from aiogym.models.registry import apply_model_params, make_model
from aiogym.models.cases import (
    case_profile_hash,
    configure_model_for_case,
    load_case,
)

from .specs import DISTRIBUTION_SCHEMA_VERSION, DistributionSpec


def distribution_spec_from_case(
    case: str | Mapping[str, Any],
    *,
    scenario: str | None = None,
    goal: str = "regulation",
) -> DistributionSpec:
    """Compile one deterministic Case v2 profile into a fixed distribution."""

    profile = load_case(case, scenario=scenario)
    selected_scenario = str(profile["scenario"])
    environment = dict(profile["environment"])
    model = apply_model_params(
        make_model(selected_scenario),
        profile.get("model_params", {}),
    )
    configure_model_for_case(model, profile)

    initial_state = copy.deepcopy(
        profile.get("initialization", {}).get(
            "state",
            model.initial_state(),
        )
    )
    setpoints = dict(profile.get("setpoints", {}))
    initial_reference = copy.deepcopy(
        setpoints.get(
            "initial",
            model.env_setpoint_vector({"randomize_setpoints": False}),
        )
    )
    reference_by_step = {
        int(event["at_step"]): {
            **copy.deepcopy(dict(event)),
            "at_step": int(event["at_step"]),
        }
        for event in setpoints.get("schedule", [])
    }
    reference_by_step.setdefault(
        0,
        {"at_step": 0, "values": initial_reference},
    )
    reference_schedule = [
        reference_by_step[step] for step in sorted(reference_by_step)
    ]

    initial_disturbances = model.runtime_env(model.disturbance_defaults())
    disturbance_by_identity = {
        (0, str(name)): {
            "at_step": 0,
            "name": str(name),
            "value": copy.deepcopy(value),
        }
        for name, value in initial_disturbances.items()
    }
    for event in profile.get("disturbances", []):
        key = (int(event["at_step"]), str(event["name"]))
        disturbance_by_identity[key] = copy.deepcopy(dict(event))
    disturbance_schedule = sorted(
        disturbance_by_identity.values(),
        key=lambda event: (int(event["at_step"]), str(event["name"])),
    )

    noise_enabled = bool(environment.get("noise", False))
    sensor_distribution = (
        {
            "kind": "additive_gaussian",
            "noise_pct": float(environment.get("noise_pct", 0.0)),
        }
        if noise_enabled
        else {"kind": "identity"}
    )
    case_id = f"{selected_scenario}/{profile['name']}"
    return DistributionSpec(
        {
            "schema_version": DISTRIBUTION_SCHEMA_VERSION,
            "distribution_id": f"fixed-case:{case_id}:v1",
            "scenario_id": selected_scenario,
            "goal": goal,
            "control_dt": float(environment["control_dt"]),
            "episode_steps": int(environment["episode_steps"]),
            "plant_distribution": {
                "kind": "fixed",
                "parameters": copy.deepcopy(model.p),
                "case_id": case_id,
                "case_profile_hash": case_profile_hash(profile),
            },
            "initial_state_distribution": {
                "kind": "fixed",
                "state": initial_state,
            },
            "reference_distribution": {
                "kind": "fixed",
                "schedule": reference_schedule,
            },
            "disturbance_distribution": {
                "kind": "fixed",
                "schedule": disturbance_schedule,
            },
            "sensor_distribution": sensor_distribution,
            "actuator_distribution": {"kind": "identity"},
            "economic_context_distribution": {
                "kind": "fixed",
                "context": copy.deepcopy(
                    getattr(model, "economic_config", {})
                ),
            },
            "mixture_weights": {"fixed": 1.0},
            "curriculum_id": None,
        }
    )

__all__ = ["distribution_spec_from_case"]

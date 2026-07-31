"""Feasible L0-L2 distributions for the recirculating cascade process."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from aiogym.models.registry import apply_model_params, make_model
from aiogym.models.core import RHO_CP

from .feasibility import (
    bounded_disturbance_value,
    correlated_equilibrium_perturbation,
    initial_disturbance_schedule,
    rejection_sample,
    require_finite_vector,
)
from .seed_tree import SeedTree
from .specs import (
    DISTRIBUTION_SCHEMA_VERSION,
    DistributionSpec,
    EpisodeSpec,
)


_LEVELS = ("L0", "L1", "L2")


def cascade_recirculating_training_distribution(
    level: str = "L2",
    *,
    control_dt: float = 0.5,
    episode_steps: int = 1600,
) -> DistributionSpec:
    """Build a versioned recirculating-cascade regulation distribution."""

    selected = _normalize_level(level)
    varying_state = selected != "L0"
    varying_plant = selected == "L2"
    return DistributionSpec(
        {
            "schema_version": DISTRIBUTION_SCHEMA_VERSION,
            "distribution_id": (
                "cascade-recirculating-regulation-training-"
                f"{selected.lower()}-v1"
            ),
            "scenario_id": "cascade-recirculating",
            "goal": "regulation",
            "control_dt": float(control_dt),
            "episode_steps": int(episode_steps),
            "plant_distribution": {
                "kind": (
                    "bounded_correlated_parameters_v1"
                    if varying_plant
                    else "fixed"
                ),
                "relative_half_width": 0.08 if varying_plant else 0.0,
                "parameters": [
                    "area",
                    "cv_interstage",
                    "ua_loss",
                    "heater_power",
                    "pump_flow_max",
                ],
            },
            "initial_state_distribution": {
                "kind": (
                    "correlated_equilibrium_perturbation"
                    if varying_state
                    else "fixed_equilibrium"
                ),
                "level_std": 0.006 if varying_state else 0.0,
                "temperature_std": 0.30 if varying_state else 0.0,
                "level_safety_margin": 0.07,
            },
            "reference_distribution": {
                "kind": (
                    "two_feasible_equilibria_v1"
                    if varying_state
                    else "fixed_hold"
                ),
                "event_count": 1 if varying_state else 0,
                "derived_outputs": [
                    "tank_2_temperature",
                    "tank_3_temperature",
                ],
            },
            "disturbance_distribution": {
                "kind": (
                    "bounded_single_shift_v1"
                    if varying_plant
                    else "fixed_nominal"
                ),
                "event_count": 1 if varying_plant else 0,
                "names": [
                    "t_amb",
                    "pump_flow_factor",
                    "heater_efficiency",
                    "heat_loss_factor",
                ],
            },
            "sensor_distribution": {"kind": "identity"},
            "actuator_distribution": {"kind": "identity"},
            "economic_context_distribution": {
                "kind": "fixed",
                "context": copy.deepcopy(
                    make_model(
                        "cascade-recirculating"
                    ).economic_config
                ),
            },
            "mixture_weights": {"equilibrium_regulation": 1.0},
            "curriculum_id": (
                "cascade-recirculating-regulation-curriculum-v1:"
                f"{selected}"
            ),
        }
    )


class CascadeRecirculatingTrainingSampler:
    """Generate deterministic episodes from model-derived equilibria."""

    def __init__(
        self,
        distribution: DistributionSpec | None = None,
        *,
        level: str = "L2",
        split: str = "training",
        control_dt: float = 0.5,
        episode_steps: int = 1600,
    ) -> None:
        self.distribution = (
            distribution
            if distribution is not None
            else cascade_recirculating_training_distribution(
                level,
                control_dt=control_dt,
                episode_steps=episode_steps,
            )
        )
        if not isinstance(self.distribution, DistributionSpec):
            raise TypeError("distribution must be a DistributionSpec")
        if self.distribution.scenario_id != "cascade-recirculating":
            raise ValueError(
                "CascadeRecirculatingTrainingSampler requires "
                "scenario_id='cascade-recirculating'"
            )
        self.split = str(split)

    @property
    def distribution_id(self) -> str:
        return self.distribution.distribution_id

    @property
    def distribution_hash(self) -> str:
        return self.distribution.distribution_hash

    def sample(
        self,
        seed: int,
        *,
        worker_index: int = 0,
        episode_index: int = 0,
    ) -> EpisodeSpec:
        tree = SeedTree.for_split(
            seed,
            self.distribution_id,
            self.split,
            worker_index=worker_index,
            episode_index=episode_index,
        )
        level = _distribution_level(self.distribution)
        plant_rng = tree.generator("plant")
        initial_rng = tree.generator("initial_state")
        reference_rng = tree.generator("reference")
        disturbance_rng = tree.generator("disturbance")
        parameters = _sample_parameters(plant_rng, level)
        model = apply_model_params(
            make_model("cascade-recirculating"),
            parameters,
        )
        initial_environment = _sample_initial_environment(
            model,
            disturbance_rng,
            level,
        )
        operating = _sample_equilibrium(
            model,
            reference_rng,
            initial_environment,
            nominal=(level == "L0"),
            component="recirculating-cascade operating point",
        )
        initial_state = correlated_equilibrium_perturbation(
            model,
            operating["state"],
            initial_rng,
            level_std=0.0 if level == "L0" else 0.006,
            temperature_std=0.0 if level == "L0" else 0.30,
            level_margin=0.07,
        )
        reference_schedule = [
            {
                "at_step": 0,
                "values": list(operating["y_sp"]),
                "kind": "initial",
            }
        ]
        if level != "L0":
            target = _sample_equilibrium(
                model,
                reference_rng,
                initial_environment,
                nominal=False,
                component="recirculating-cascade reference event",
                separated_from=list(operating["y_sp"]),
            )
            reference_schedule.append(
                {
                    "at_step": _event_step(
                        reference_rng,
                        self.distribution.episode_steps,
                    ),
                    "values": list(target["y_sp"]),
                    "kind": "feasible_step",
                }
            )
        disturbance_schedule = _sample_disturbances(
            model,
            disturbance_rng,
            initial_environment,
            level,
            self.distribution.episode_steps,
        )
        declaration = self.distribution.declaration
        episode = EpisodeSpec.from_distribution(
            self.distribution,
            base_seed=tree.base_seed,
            component_seeds=tree.component_seeds,
            plant_parameters=parameters,
            initial_state=initial_state,
            reference_schedule=sorted(
                reference_schedule,
                key=lambda event: int(event["at_step"]),
            ),
            disturbance_schedule=disturbance_schedule,
            sensor_model=declaration["sensor_distribution"],
            actuator_model=declaration["actuator_distribution"],
            economic_context=declaration[
                "economic_context_distribution"
            ].get("context", {}),
            difficulty_tags=(level, "model-derived-equilibrium"),
        )
        report = validate_cascade_recirculating_episode(episode)
        if not report["passed"]:
            failures = ", ".join(
                check["name"]
                for check in report["checks"]
                if not check["passed"]
            )
            raise RuntimeError(
                "generated infeasible recirculating-cascade episode: "
                + failures
            )
        return episode


def validate_cascade_recirculating_episode(
    episode: EpisodeSpec,
) -> dict[str, Any]:
    """Validate bounds and model consistency for a resolved episode."""

    if not isinstance(episode, EpisodeSpec):
        raise TypeError("episode must be an EpisodeSpec")
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "detail": str(detail)}
        )

    try:
        model = apply_model_params(
            make_model("cascade-recirculating"),
            episode.plant_parameters,
        )
        record("parameter_snapshot", True)
    except (KeyError, TypeError, ValueError) as exc:
        record("parameter_snapshot", False, str(exc))
        return {"passed": False, "checks": checks}

    state = require_finite_vector(
        "initial state",
        episode.initial_state,
        expected_length=6,
    )
    state_ok = all(
        _within(value, row.get("bounds"))
        for value, row in zip(state, model.state_schema())
    )
    record("initial_state_bounds", state_ok)

    environment = model.runtime_env(model.disturbance_defaults())
    disturbances_ok = True
    disturbance_detail = ""
    try:
        for event in episode.disturbance_schedule:
            environment[str(event["name"])] = event["value"]
            model.runtime_env(environment)
    except (KeyError, TypeError, ValueError) as exc:
        disturbances_ok = False
        disturbance_detail = str(exc)
    record("disturbance_bounds", disturbances_ok, disturbance_detail)

    initial_environment = model.runtime_env(model.disturbance_defaults())
    for event in episode.disturbance_schedule:
        if int(event["at_step"]) == 0:
            initial_environment[str(event["name"])] = event["value"]
    references_ok = True
    reference_details: list[str] = []
    for event in episode.reference_schedule:
        try:
            equilibrium = _equilibrium_from_reference(
                model,
                event["values"],
                initial_environment,
            )
            if not equilibrium["feasible"]:
                references_ok = False
                reference_details.append(
                    repr(equilibrium["infeasible_reasons"])
                )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            references_ok = False
            reference_details.append(str(exc))
    record(
        "reference_feasibility",
        references_ok,
        "; ".join(reference_details),
    )

    event_times_ok = all(
        0 <= int(event["at_step"]) < episode.episode_steps
        for event in (
            *episode.reference_schedule,
            *episode.disturbance_schedule,
        )
    )
    record("event_times", event_times_ok)
    record(
        "finite_payload",
        all(math.isfinite(value) for value in state),
    )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _normalize_level(level: str) -> str:
    selected = str(level).upper()
    if selected not in _LEVELS:
        raise ValueError(
            "recirculating cascade training level must be one of: "
            "L0, L1, L2"
        )
    return selected


def _distribution_level(distribution: DistributionSpec) -> str:
    curriculum = distribution.declaration.get("curriculum_id")
    if not curriculum:
        raise ValueError(
            "recirculating cascade distribution must declare curriculum_id"
        )
    return _normalize_level(str(curriculum).rsplit(":", 1)[-1])


def _sample_parameters(
    rng: np.random.Generator,
    level: str,
) -> dict[str, Any]:
    model = make_model("cascade-recirculating")
    parameters = copy.deepcopy(model.p)
    if level != "L2":
        return parameters

    geometry = float(rng.uniform(0.96, 1.04))
    hydraulic = float(rng.uniform(0.94, 1.06))
    thermal = float(rng.uniform(0.94, 1.06))
    parameters["area"] = [
        float(value * geometry * rng.uniform(0.985, 1.015))
        for value in parameters["area"]
    ]
    parameters["cv_interstage"] = [
        float(value * hydraulic * rng.uniform(0.985, 1.015))
        for value in parameters["cv_interstage"]
    ]
    parameters["ua_loss"] = [
        float(value * thermal * rng.uniform(0.985, 1.015))
        for value in parameters["ua_loss"]
    ]
    parameters["pump_flow_max"] *= hydraulic
    parameters["heater_power"] *= float(rng.uniform(0.94, 1.0))
    return parameters


def _sample_initial_environment(
    model,
    rng: np.random.Generator,
    level: str,
) -> dict[str, float]:
    defaults = model.runtime_env(model.disturbance_defaults())
    if level != "L2":
        return defaults
    defaults.update(
        {
            "t_amb": float(rng.uniform(18.0, 22.0)),
            "pump_flow_factor": float(rng.uniform(0.92, 1.08)),
            "heater_efficiency": float(rng.uniform(0.90, 1.0)),
            "heat_loss_factor": float(rng.uniform(0.88, 1.12)),
        }
    )
    return model.runtime_env(defaults)


def _sample_equilibrium(
    model,
    rng: np.random.Generator,
    environment: Mapping[str, float],
    *,
    nominal: bool,
    component: str,
    separated_from: list[float] | None = None,
) -> Mapping[str, Any]:
    if nominal:
        result = model.nominal_steady_state(env=environment)
        if not result["feasible"]:
            raise RuntimeError(
                "nominal recirculating-cascade equilibrium is infeasible: "
                + repr(result["infeasible_reasons"])
            )
        return result

    def candidate() -> dict[str, Any]:
        return {
            "circulation_flow": float(rng.uniform(6.5e-5, 9.0e-5)),
            "tank_1_temperature": float(rng.uniform(27.0, 35.0)),
            "levels": [
                float(value)
                for value in rng.uniform(0.34, 0.42, size=3)
            ],
        }

    def feasible(values: Mapping[str, Any]) -> Mapping[str, Any]:
        result = model.nominal_steady_state(
            **values,
            env=environment,
        )
        if result["feasible"] and separated_from is not None:
            difference = max(
                abs(float(left) - float(right))
                for left, right in zip(result["y_sp"], separated_from)
            )
            if difference < 0.8:
                return {
                    "feasible": False,
                    "infeasible_reasons": (
                        "reference equilibrium lacks separation"
                    ),
                }
        return result

    values = rejection_sample(
        component,
        candidate,
        feasible,
        max_attempts=64,
    )
    return model.nominal_steady_state(**values, env=environment)


def _equilibrium_from_reference(
    model,
    reference,
    environment: Mapping[str, float],
) -> Mapping[str, Any]:
    values = require_finite_vector(
        "reference",
        reference,
        expected_length=6,
    )
    levels = values[:3]
    t1, t2, _ = values[3:]
    ambient = float(environment["t_amb"])
    ua2 = (
        float(model.p["ua_loss"][1])
        * float(environment["heat_loss_factor"])
    )
    denominator = RHO_CP * (t1 - t2)
    if denominator <= 0.0:
        raise ValueError(
            "recirculating reference must have tank_1_temperature "
            "above tank_2_temperature"
        )
    flow = ua2 * (t2 - ambient) / denominator
    equilibrium = model.nominal_steady_state(
        circulation_flow=flow,
        tank_1_temperature=t1,
        levels=levels,
        env=environment,
    )
    error = max(
        abs(float(left) - float(right))
        for left, right in zip(equilibrium["y_sp"], values)
    )
    if error > 1e-7:
        return {
            **equilibrium,
            "feasible": False,
            "infeasible_reasons": (
                *equilibrium["infeasible_reasons"],
                f"reference differs from derived equilibrium by {error}",
            ),
        }
    return equilibrium


def _event_step(
    rng: np.random.Generator,
    episode_steps: int,
) -> int:
    lower = max(1, int(0.20 * episode_steps))
    upper = max(lower + 1, int(0.70 * episode_steps))
    return int(rng.integers(lower, min(upper, episode_steps)))


def _sample_disturbances(
    model,
    rng: np.random.Generator,
    initial_environment: Mapping[str, float],
    level: str,
    episode_steps: int,
) -> list[dict[str, Any]]:
    events = initial_disturbance_schedule(model, initial_environment)
    if level != "L2":
        return events
    names = (
        "t_amb",
        "pump_flow_factor",
        "heater_efficiency",
        "heat_loss_factor",
    )
    name = names[int(rng.integers(0, len(names)))]
    initial = float(initial_environment[name])
    if name == "t_amb":
        shifted = initial + float(rng.uniform(-2.5, 2.5))
    elif name == "heater_efficiency":
        shifted = initial * float(rng.uniform(0.82, 0.96))
    else:
        shifted = initial * float(rng.uniform(0.85, 1.15))
    events.append(
        {
            "at_step": _event_step(rng, episode_steps),
            "name": name,
            "value": bounded_disturbance_value(model, name, shifted),
            "kind": "bounded_shift",
            "observability": "unmeasured",
        }
    )
    return sorted(
        events,
        key=lambda event: (int(event["at_step"]), str(event["name"])),
    )


def _within(value: float, bounds) -> bool:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        return True
    lower, upper = bounds
    return (
        (lower is None or value >= float(lower))
        and (upper is None or value <= float(upper))
    )


__all__ = [
    "CascadeRecirculatingTrainingSampler",
    "cascade_recirculating_training_distribution",
    "validate_cascade_recirculating_episode",
]

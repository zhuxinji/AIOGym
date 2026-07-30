"""Feasible L0-L2 training distributions for the open cascade process."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from aiogym.models import apply_model_params, make_model

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


CASCADE_PRODUCT_FLOW_M3S = 4.0e-4
_LEVELS = ("L0", "L1", "L2")


def cascade_training_distribution(
    level: str = "L2",
    *,
    control_dt: float = 0.5,
    episode_steps: int = 1600,
) -> DistributionSpec:
    """Build one versioned open-cascade regulation distribution."""

    selected = _normalize_level(level)
    varying_state = selected != "L0"
    varying_plant = selected == "L2"
    return DistributionSpec(
        {
            "schema_version": DISTRIBUTION_SCHEMA_VERSION,
            "distribution_id": (
                f"cascade-regulation-training-{selected.lower()}-v1"
            ),
            "scenario_id": "cascade",
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
                    "cv_out",
                    "ua_loss",
                    "heater_max",
                    "pump_flow_max",
                ],
                "product_flow_sp_m3s": CASCADE_PRODUCT_FLOW_M3S,
            },
            "initial_state_distribution": {
                "kind": (
                    "correlated_equilibrium_perturbation"
                    if varying_state
                    else "fixed_equilibrium"
                ),
                "level_std": 0.012 if varying_state else 0.0,
                "temperature_std": 0.45 if varying_state else 0.0,
                "level_safety_margin": 0.08,
            },
            "reference_distribution": {
                "kind": (
                    "feasible_equilibrium_step"
                    if varying_state
                    else "fixed_hold"
                ),
                "event_count": 1 if varying_state else 0,
                "minimum_dwell_fraction": 0.20,
            },
            "disturbance_distribution": {
                "kind": (
                    "bounded_single_shift_v1"
                    if varying_plant
                    else "fixed_nominal"
                ),
                "event_count": 1 if varying_plant else 0,
                "names": [
                    "t_cold",
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
                "context": copy.deepcopy(make_model("cascade").economic_config),
            },
            "mixture_weights": {"equilibrium_regulation": 1.0},
            "curriculum_id": f"cascade-regulation-curriculum-v1:{selected}",
        }
    )


class CascadeTrainingSampler:
    """Generate deterministic, feasible open-cascade episodes."""

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
            else cascade_training_distribution(
                level,
                control_dt=control_dt,
                episode_steps=episode_steps,
            )
        )
        if not isinstance(self.distribution, DistributionSpec):
            raise TypeError("distribution must be a DistributionSpec")
        if self.distribution.scenario_id != "cascade":
            raise ValueError(
                "CascadeTrainingSampler requires scenario_id='cascade'"
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
        model = apply_model_params(make_model("cascade"), parameters)
        model.configure_operation(
            {
                "product_flow_sp": CASCADE_PRODUCT_FLOW_M3S,
                "min_product_flow": CASCADE_PRODUCT_FLOW_M3S,
            }
        )
        initial_environment = _sample_initial_environment(
            model,
            disturbance_rng,
            level,
        )
        operating_reference, _ = _sample_feasible_reference(
            model,
            reference_rng,
            initial_environment,
            level=level,
            nominal=(level == "L0"),
            component="cascade operating reference",
        )
        equilibrium = _reference_to_state(operating_reference)
        initial_state = correlated_equilibrium_perturbation(
            model,
            equilibrium,
            initial_rng,
            level_std=0.0 if level == "L0" else 0.012,
            temperature_std=0.0 if level == "L0" else 0.45,
            level_margin=0.08,
        )
        reference_schedule = [
            {
                "at_step": 0,
                "values": operating_reference,
                "kind": "initial",
            }
        ]
        if level != "L0":
            target, _ = _sample_feasible_reference(
                model,
                reference_rng,
                initial_environment,
                level=level,
                nominal=False,
                component="cascade reference event",
                separated_from=operating_reference,
            )
            reference_schedule.append(
                {
                    "at_step": _event_step(
                        reference_rng,
                        self.distribution.episode_steps,
                    ),
                    "values": target,
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
            difficulty_tags=(level, "feasible-equilibrium"),
        )
        report = validate_cascade_episode(episode)
        if not report["passed"]:
            failures = ", ".join(
                check["name"]
                for check in report["checks"]
                if not check["passed"]
            )
            raise RuntimeError(
                "generated infeasible cascade episode: " + failures
            )
        return episode


def validate_cascade_episode(episode: EpisodeSpec) -> dict[str, Any]:
    """Validate physical bounds and equilibrium references for one episode."""

    if not isinstance(episode, EpisodeSpec):
        raise TypeError("episode must be an EpisodeSpec")
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "detail": str(detail)}
        )

    try:
        model = apply_model_params(
            make_model("cascade"),
            episode.plant_parameters,
        )
        model.configure_operation(
            {
                "product_flow_sp": CASCADE_PRODUCT_FLOW_M3S,
                "min_product_flow": CASCADE_PRODUCT_FLOW_M3S,
            }
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
    reference_ok = True
    reasons: list[str] = []
    for event in episode.reference_schedule:
        try:
            values = require_finite_vector(
                "reference",
                event["values"],
                expected_length=6,
            )
            result = model.steady_state_requirements(
                values,
                env=initial_environment,
            )
            if not result["feasible"]:
                reference_ok = False
                reasons.append(repr(result["infeasible_reasons"]))
        except (KeyError, TypeError, ValueError) as exc:
            reference_ok = False
            reasons.append(str(exc))
    record("reference_feasibility", reference_ok, "; ".join(reasons))

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
        raise ValueError("cascade training level must be one of: L0, L1, L2")
    return selected


def _distribution_level(distribution: DistributionSpec) -> str:
    curriculum = distribution.declaration.get("curriculum_id")
    if not curriculum:
        raise ValueError("cascade distribution must declare curriculum_id")
    return _normalize_level(str(curriculum).rsplit(":", 1)[-1])


def _sample_parameters(
    rng: np.random.Generator,
    level: str,
) -> dict[str, Any]:
    model = make_model("cascade")
    parameters = copy.deepcopy(model.p)
    if level != "L2":
        return parameters
    shared_hydraulic = float(rng.uniform(0.94, 1.06))
    shared_thermal = float(rng.uniform(0.94, 1.06))
    parameters["area"] *= float(rng.uniform(0.96, 1.04))
    parameters["cv_out"] *= shared_hydraulic
    parameters["pump_flow_max"] *= shared_hydraulic
    parameters["ua_loss"] *= shared_thermal
    parameters["heater_max"] *= float(rng.uniform(0.94, 1.06))
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
            "t_cold": float(rng.uniform(13.5, 16.5)),
            "t_amb": float(rng.uniform(18.0, 22.0)),
            "pump_flow_factor": float(rng.uniform(0.92, 1.08)),
            "heater_efficiency": float(rng.uniform(0.88, 1.0)),
            "heat_loss_factor": float(rng.uniform(0.85, 1.15)),
            "extra_outflow": 0.0,
        }
    )
    return model.runtime_env(defaults)


def _sample_feasible_reference(
    model,
    rng: np.random.Generator,
    environment: Mapping[str, float],
    *,
    level: str,
    nominal: bool,
    component: str,
    separated_from: list[float] | None = None,
) -> tuple[list[float], Mapping[str, Any]]:
    if nominal:
        reference = list(model.default_y_sp)
        result = model.steady_state_requirements(reference, env=environment)
        if not result["feasible"]:
            raise RuntimeError(
                "nominal cascade reference is infeasible: "
                + repr(result["infeasible_reasons"])
            )
        return reference, result

    def candidate() -> list[float]:
        levels = rng.uniform(0.36, 0.54, size=3)
        t0 = float(rng.uniform(28.0, 40.0))
        t1 = t0 + float(rng.uniform(7.0, 14.0))
        t2 = t1 + float(rng.uniform(7.0, 14.0))
        return [
            *(float(value) for value in levels),
            t0,
            t1,
            t2,
        ]

    def feasible(values: list[float]) -> Mapping[str, Any]:
        result = model.steady_state_requirements(values, env=environment)
        if result["feasible"] and separated_from is not None:
            difference = max(
                abs(float(left) - float(right))
                for left, right in zip(values, separated_from)
            )
            if difference < 1.0:
                return {
                    "feasible": False,
                    "infeasible_reasons": "reference event lacks separation",
                }
        return result

    reference = rejection_sample(
        component,
        candidate,
        feasible,
        max_attempts=64,
    )
    return reference, model.steady_state_requirements(
        reference,
        env=environment,
    )


def _reference_to_state(reference: list[float]) -> list[float]:
    levels = reference[:3]
    temperatures = reference[3:]
    return [
        levels[0],
        temperatures[0],
        levels[1],
        temperatures[1],
        levels[2],
        temperatures[2],
    ]


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
    name = (
        "t_cold",
        "t_amb",
        "pump_flow_factor",
        "heater_efficiency",
        "heat_loss_factor",
    )[int(rng.integers(0, 5))]
    initial = float(initial_environment[name])
    if name in {"t_cold", "t_amb"}:
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
    "CASCADE_PRODUCT_FLOW_M3S",
    "CascadeTrainingSampler",
    "cascade_training_distribution",
    "validate_cascade_episode",
]

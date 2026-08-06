"""Strict, versioned input contract for three-tank design studies."""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiogym.core import stable_hash


DESIGN_SPEC_SCHEMA_VERSION = "aiogym.design_spec.v1"
DESIGN_TOPOLOGY = "three-tank-recirculating"

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "description",
        "topology",
        "tanks",
        "heaters",
        "pump",
        "hydraulics",
        "operation",
        "requirements",
        "uncertainties",
        "references",
    }
)
_TANK_FIELDS = frozenset(
    {
        "area_m2",
        "height_m",
        "level_sensor_range_m",
        "ua_w_per_k",
        "low_level_trip_m",
        "high_level_trip_m",
        "overflow_level_m",
    }
)
_HEATER_FIELDS = frozenset({"tank", "power_w", "efficiency"})
_PUMP_FIELDS = frozenset(
    {
        "motor_power_w",
        "max_flow_m3s",
        "static_head_m",
        "shutoff_head_m",
    }
)
_HYDRAULIC_FIELDS = frozenset(
    {
        "cv_interstage",
        "gravity_drop_m",
        "cv_overflow",
        "overflow_head_floor_m",
    }
)
_OPERATION_FIELDS = frozenset(
    {
        "circulation_flow_m3s",
        "target_levels_m",
        "target_temperatures_degC",
        "initial_levels_m",
        "initial_temperatures_degC",
        "ambient_temperature_degC",
        "control_dt_s",
        "duration_s",
    }
)
_REQUIREMENT_FIELDS = frozenset(
    {
        "maximum_heatup_time_s",
        "temperature_tolerance_degC",
        "level_tolerance_m",
        "maximum_overshoot_degC",
        "maximum_energy_kwh",
        "minimum_actuator_margin",
        "temperature_trip_degC",
        "temperature_hard_limit_degC",
        "robustness_pass_rate",
    }
)
_UNCERTAINTY_FIELDS = frozenset(
    {
        "samples",
        "seed",
        "pump_flow_factor",
        "heater_efficiency_factor",
        "heat_loss_factor",
    }
)


def load_design_spec(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load, validate, normalize, and hash one design declaration."""

    if isinstance(source, Mapping):
        raw = copy.deepcopy(dict(source))
        raw.pop("design_hash", None)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"design spec not found: {source}")
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    resolved = validate_design_spec(raw)
    resolved["design_hash"] = stable_hash(resolved)
    return resolved


def validate_design_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical design spec or raise a precise validation error."""

    if not isinstance(spec, Mapping):
        raise TypeError("design spec must be a mapping")
    unknown = set(spec) - _ROOT_FIELDS
    if unknown:
        raise ValueError("unknown design fields: " + ", ".join(sorted(unknown)))
    required = {
        "schema_version",
        "id",
        "topology",
        "tanks",
        "heaters",
        "pump",
        "hydraulics",
        "operation",
        "requirements",
    }
    missing = required - set(spec)
    if missing:
        raise ValueError("design spec is missing fields: " + ", ".join(sorted(missing)))
    if spec["schema_version"] != DESIGN_SPEC_SCHEMA_VERSION:
        raise ValueError(f"unsupported design schema: {spec['schema_version']!r}")
    design_id = _nonempty_string("id", spec["id"])
    if spec["topology"] != DESIGN_TOPOLOGY:
        raise ValueError(f"topology must be {DESIGN_TOPOLOGY!r} in design_spec.v1")

    tanks = _sequence("tanks", spec["tanks"], length=3)
    normalized_tanks = []
    for index, raw_tank in enumerate(tanks):
        tank = _mapping(f"tanks[{index}]", raw_tank, _TANK_FIELDS)
        required_tank = _TANK_FIELDS - ({"overflow_level_m"} if index == 2 else set())
        _require_fields(f"tanks[{index}]", tank, required_tank)
        row = {
            "area_m2": _positive(f"tanks[{index}].area_m2", tank["area_m2"]),
            "height_m": _positive(f"tanks[{index}].height_m", tank["height_m"]),
            "level_sensor_range_m": _positive(
                f"tanks[{index}].level_sensor_range_m", tank["level_sensor_range_m"]
            ),
            "ua_w_per_k": _nonnegative(
                f"tanks[{index}].ua_w_per_k", tank["ua_w_per_k"]
            ),
            "low_level_trip_m": _nonnegative(
                f"tanks[{index}].low_level_trip_m", tank["low_level_trip_m"]
            ),
            "high_level_trip_m": _positive(
                f"tanks[{index}].high_level_trip_m", tank["high_level_trip_m"]
            ),
        }
        if index < 2:
            row["overflow_level_m"] = _positive(
                f"tanks[{index}].overflow_level_m", tank["overflow_level_m"]
            )
        normalized_tanks.append(row)

    heaters = _sequence("heaters", spec["heaters"], maximum=3)
    normalized_heaters = []
    seen_tanks = set()
    for index, raw_heater in enumerate(heaters):
        heater = _mapping(f"heaters[{index}]", raw_heater, _HEATER_FIELDS)
        _require_fields(f"heaters[{index}]", heater, {"tank", "power_w"})
        tank_number = heater["tank"]
        if isinstance(tank_number, bool) or not isinstance(tank_number, int):
            raise TypeError(f"heaters[{index}].tank must be an integer")
        if tank_number not in (1, 2, 3):
            raise ValueError(f"heaters[{index}].tank must be 1, 2, or 3")
        if tank_number in seen_tanks:
            raise ValueError(f"only one heater slot may be declared for Tank {tank_number}")
        seen_tanks.add(tank_number)
        normalized_heaters.append(
            {
                "tank": tank_number,
                "power_w": _positive(f"heaters[{index}].power_w", heater["power_w"]),
                "efficiency": _bounded(
                    f"heaters[{index}].efficiency",
                    heater.get("efficiency", 1.0),
                    0.0,
                    1.0,
                    lower_open=True,
                ),
            }
        )
    normalized_heaters.sort(key=lambda row: row["tank"])

    pump = _mapping("pump", spec["pump"], _PUMP_FIELDS)
    _require_fields("pump", pump, _PUMP_FIELDS)
    normalized_pump = {
        name: _positive(f"pump.{name}", pump[name])
        for name in _PUMP_FIELDS
    }
    hydraulics = _mapping("hydraulics", spec["hydraulics"], _HYDRAULIC_FIELDS)
    _require_fields(
        "hydraulics",
        hydraulics,
        {"cv_interstage", "gravity_drop_m", "cv_overflow"},
    )
    normalized_hydraulics = {
        "cv_interstage": _numeric_vector(
            "hydraulics.cv_interstage", hydraulics["cv_interstage"], 2, positive=True
        ),
        "gravity_drop_m": _numeric_vector(
            "hydraulics.gravity_drop_m", hydraulics["gravity_drop_m"], 2, positive=True
        ),
        "cv_overflow": _numeric_vector(
            "hydraulics.cv_overflow", hydraulics["cv_overflow"], 2, positive=True
        ),
        "overflow_head_floor_m": _positive(
            "hydraulics.overflow_head_floor_m",
            hydraulics.get("overflow_head_floor_m", 1e-9),
        ),
    }

    operation = _mapping("operation", spec["operation"], _OPERATION_FIELDS)
    _require_fields("operation", operation, _OPERATION_FIELDS)
    normalized_operation = {
        "circulation_flow_m3s": _positive(
            "operation.circulation_flow_m3s", operation["circulation_flow_m3s"]
        ),
        "target_levels_m": _numeric_vector(
            "operation.target_levels_m", operation["target_levels_m"], 3, positive=True
        ),
        "target_temperatures_degC": _numeric_vector(
            "operation.target_temperatures_degC",
            operation["target_temperatures_degC"],
            3,
        ),
        "initial_levels_m": _numeric_vector(
            "operation.initial_levels_m", operation["initial_levels_m"], 3, positive=True
        ),
        "initial_temperatures_degC": _numeric_vector(
            "operation.initial_temperatures_degC",
            operation["initial_temperatures_degC"],
            3,
        ),
        "ambient_temperature_degC": _finite(
            "operation.ambient_temperature_degC", operation["ambient_temperature_degC"]
        ),
        "control_dt_s": _positive("operation.control_dt_s", operation["control_dt_s"]),
        "duration_s": _positive("operation.duration_s", operation["duration_s"]),
    }

    requirements = _mapping(
        "requirements", spec["requirements"], _REQUIREMENT_FIELDS
    )
    _require_fields("requirements", requirements, _REQUIREMENT_FIELDS)
    normalized_requirements = {
        "maximum_heatup_time_s": _positive(
            "requirements.maximum_heatup_time_s", requirements["maximum_heatup_time_s"]
        ),
        "temperature_tolerance_degC": _positive(
            "requirements.temperature_tolerance_degC",
            requirements["temperature_tolerance_degC"],
        ),
        "level_tolerance_m": _positive(
            "requirements.level_tolerance_m", requirements["level_tolerance_m"]
        ),
        "maximum_overshoot_degC": _nonnegative(
            "requirements.maximum_overshoot_degC",
            requirements["maximum_overshoot_degC"],
        ),
        "maximum_energy_kwh": _positive(
            "requirements.maximum_energy_kwh", requirements["maximum_energy_kwh"]
        ),
        "minimum_actuator_margin": _bounded(
            "requirements.minimum_actuator_margin",
            requirements["minimum_actuator_margin"],
            0.0,
            1.0,
        ),
        "temperature_trip_degC": _finite(
            "requirements.temperature_trip_degC", requirements["temperature_trip_degC"]
        ),
        "temperature_hard_limit_degC": _finite(
            "requirements.temperature_hard_limit_degC",
            requirements["temperature_hard_limit_degC"],
        ),
        "robustness_pass_rate": _bounded(
            "requirements.robustness_pass_rate",
            requirements["robustness_pass_rate"],
            0.0,
            1.0,
        ),
    }

    uncertainty = _mapping(
        "uncertainties", spec.get("uncertainties", {}), _UNCERTAINTY_FIELDS
    )
    samples = uncertainty.get("samples", 12)
    seed = uncertainty.get("seed", 0)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("uncertainties.samples must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("uncertainties.seed must be a non-negative integer")
    normalized_uncertainty = {
        "samples": samples,
        "seed": seed,
        "pump_flow_factor": _range(
            "uncertainties.pump_flow_factor",
            uncertainty.get("pump_flow_factor", [0.8, 1.0]),
            positive=True,
        ),
        "heater_efficiency_factor": _range(
            "uncertainties.heater_efficiency_factor",
            uncertainty.get("heater_efficiency_factor", [0.8, 1.0]),
            positive=True,
            upper=1.0,
        ),
        "heat_loss_factor": _range(
            "uncertainties.heat_loss_factor",
            uncertainty.get("heat_loss_factor", [1.0, 1.5]),
            positive=True,
        ),
    }

    references = spec.get("references", [])
    if not isinstance(references, list):
        raise TypeError("references must be a list")
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise TypeError(f"references[{index}] must be a mapping")

    resolved = {
        "schema_version": DESIGN_SPEC_SCHEMA_VERSION,
        "id": design_id,
        "description": str(spec.get("description", "")),
        "topology": DESIGN_TOPOLOGY,
        "tanks": normalized_tanks,
        "heaters": normalized_heaters,
        "pump": normalized_pump,
        "hydraulics": normalized_hydraulics,
        "operation": normalized_operation,
        "requirements": normalized_requirements,
        "uncertainties": normalized_uncertainty,
        "references": copy.deepcopy(references),
    }
    _validate_cross_field_constraints(resolved)
    return resolved


def _validate_cross_field_constraints(spec: Mapping[str, Any]) -> None:
    for index, tank in enumerate(spec["tanks"]):
        label = f"Tank {index + 1}"
        if tank["level_sensor_range_m"] < tank["height_m"]:
            raise ValueError(f"{label} level sensor range must cover vessel height")
        if not tank["low_level_trip_m"] < tank["high_level_trip_m"] < tank["height_m"]:
            raise ValueError(f"{label} requires low trip < high trip < vessel height")
        if index < 2 and not (
            tank["high_level_trip_m"]
            < tank["overflow_level_m"]
            < tank["height_m"]
        ):
            raise ValueError(f"{label} requires high trip < overflow < vessel height")
    pump = spec["pump"]
    if pump["shutoff_head_m"] <= pump["static_head_m"]:
        raise ValueError("pump.shutoff_head_m must exceed pump.static_head_m")
    requirements = spec["requirements"]
    if requirements["temperature_hard_limit_degC"] <= requirements["temperature_trip_degC"]:
        raise ValueError("temperature hard limit must exceed the trip temperature")
    operation = spec["operation"]
    if operation["duration_s"] < operation["control_dt_s"]:
        raise ValueError("operation.duration_s must be at least one control interval")
    if requirements["maximum_heatup_time_s"] > operation["duration_s"]:
        raise ValueError(
            "maximum heat-up time must not exceed the simulated operation duration"
        )
    for name in ("target_levels_m", "initial_levels_m"):
        for index, level in enumerate(spec["operation"][name]):
            tank = spec["tanks"][index]
            if not tank["low_level_trip_m"] < level < tank["high_level_trip_m"]:
                raise ValueError(
                    f"operation.{name}[{index}] must lie between the low and high trips"
                )
    for name in ("target_temperatures_degC", "initial_temperatures_degC"):
        values = operation[name]
        if any(value >= requirements["temperature_hard_limit_degC"] for value in values):
            raise ValueError(f"operation.{name} must stay below the temperature hard limit")
    if any(
        value >= requirements["temperature_trip_degC"]
        for value in operation["target_temperatures_degC"]
    ):
        raise ValueError("target temperatures must stay below the heater trip temperature")


def _mapping(name, value, allowed):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ", ".join(sorted(unknown)))
    return dict(value)


def _require_fields(name, value, required):
    missing = set(required) - set(value)
    if missing:
        raise ValueError(f"{name} is missing fields: " + ", ".join(sorted(missing)))


def _sequence(name, value, *, length=None, maximum=None):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    values = list(value)
    if length is not None and len(values) != length:
        raise ValueError(f"{name} must contain {length} items")
    if maximum is not None and len(values) > maximum:
        raise ValueError(f"{name} may contain at most {maximum} items")
    return values


def _nonempty_string(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(name, value):
    number = _finite(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative(name, value):
    number = _finite(name, value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _bounded(name, value, lower, upper, *, lower_open=False):
    number = _finite(name, value)
    lower_ok = number > lower if lower_open else number >= lower
    if not lower_ok or number > upper:
        bracket = "(" if lower_open else "["
        raise ValueError(f"{name} must be in {bracket}{lower}, {upper}]")
    return number


def _numeric_vector(name, value, length, *, positive=False):
    values = _sequence(name, value, length=length)
    validator = _positive if positive else _finite
    return [validator(f"{name}[{index}]", item) for index, item in enumerate(values)]


def _range(name, value, *, positive=False, upper=None):
    values = _numeric_vector(name, value, 2, positive=positive)
    if values[0] > values[1]:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    if upper is not None and values[1] > upper:
        raise ValueError(f"{name} upper bound must be <= {upper}")
    return values


__all__ = [
    "DESIGN_SPEC_SCHEMA_VERSION",
    "DESIGN_TOPOLOGY",
    "load_design_spec",
    "validate_design_spec",
]

"""Interactive DesignSpec builder using familiar engineering units."""
from __future__ import annotations

import copy
from collections.abc import Callable

from .spec import DESIGN_SPEC_SCHEMA_VERSION, validate_design_spec


def prompt_design_spec(
    name: str,
    *,
    advanced: bool = False,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> dict:
    """Prompt for a design and return a validated declaration.

    The public DesignSpec stores SI units. The wizard accepts millimetres,
    litres per minute, kilowatts, minutes, and percentages, then converts them
    before validation.
    """

    ask = input if input_fn is None else input_fn
    tell = print if output_fn is None else output_fn
    if not isinstance(name, str) or not name.strip():
        raise ValueError("design name must be a non-empty string")
    name = name.strip()
    tell("AIO-Gym three-tank design wizard")
    tell("Press Enter to accept the value shown in [brackets].")

    default_dimensions = ([300.0, 250.0, 400.0],) * 3
    dimensions = [
        _prompt(
            ask,
            tell,
            f"Tank {index + 1} dimensions L,W,H (mm)",
            list(default_dimensions[index]),
            lambda raw: _csv_numbers(raw, 3, positive=True),
            _format_csv,
        )
        for index in range(3)
    ]
    heaters = _prompt(
        ask,
        tell,
        "Heaters tank:power_kW:efficiency_pct (comma separated; 'none' for zero)",
        [
            {"tank": 1, "power_w": 2000.0, "efficiency": 0.9},
            {"tank": 2, "power_w": 500.0, "efficiency": 0.9},
            {"tank": 3, "power_w": 500.0, "efficiency": 0.9},
        ],
        _parse_heaters,
        _format_heaters,
    )
    pump_values = _prompt(
        ask,
        tell,
        "Pump motor_kW,max_flow_Lmin,static_head_m,shutoff_head_m",
        [0.37, 25.0, 1.7, 10.0],
        lambda raw: _csv_numbers(raw, 4, positive=True),
        _format_csv,
    )
    circulation_lmin = _prompt_number(
        ask, tell, "Target circulation (L/min)", 5.0, positive=True
    )
    default_levels_mm = [0.6 * row[2] for row in dimensions]
    target_levels_mm = _prompt(
        ask,
        tell,
        "Target levels Tank1,Tank2,Tank3 (mm)",
        default_levels_mm,
        lambda raw: _csv_numbers(raw, 3, positive=True),
        _format_csv,
    )
    target_temperatures = _prompt(
        ask,
        tell,
        "Target temperatures Tank1,Tank2,Tank3 (degC)",
        [30.0, 28.97128161165881, 27.654664660905787],
        lambda raw: _csv_numbers(raw, 3),
        _format_csv,
    )
    initial_temperatures = _prompt(
        ask,
        tell,
        "Initial temperatures Tank1,Tank2,Tank3 (degC)",
        [20.0, 20.0, 20.0],
        lambda raw: _csv_numbers(raw, 3),
        _format_csv,
    )
    maximum_heatup_minutes = _prompt_number(
        ask, tell, "Maximum acceptable heat-up time (minutes)", 45.0, positive=True
    )
    maximum_energy_kwh = _prompt_number(
        ask, tell, "Maximum energy for the simulated run (kWh)", 2.5, positive=True
    )
    robustness_pass_pct = _prompt_number(
        ask,
        tell,
        "Required robustness pass rate (percent)",
        80.0,
        lower=0.0,
        upper=100.0,
    )
    robustness_samples = _prompt_integer(
        ask, tell, "Robustness samples", 12, minimum=0
    )

    tanks = []
    for index, (length_mm, width_mm, height_mm) in enumerate(dimensions):
        height_m = height_mm / 1000.0
        row = {
            "area_m2": length_mm * width_mm / 1_000_000.0,
            "height_m": height_m,
            "level_sensor_range_m": 1.25 * height_m,
            "ua_w_per_k": (40.0, 40.0, 60.0)[index],
            "low_level_trip_m": 0.20 * height_m,
            "high_level_trip_m": 0.85 * height_m,
        }
        if index < 2:
            row["overflow_level_m"] = 0.90 * height_m
        tanks.append(row)

    declaration = {
        "schema_version": DESIGN_SPEC_SCHEMA_VERSION,
        "id": name,
        "description": "Created with the AIO-Gym interactive design wizard.",
        "topology": "three-tank-recirculating",
        "tanks": tanks,
        "heaters": heaters,
        "pump": {
            "motor_power_w": pump_values[0] * 1000.0,
            "max_flow_m3s": pump_values[1] / 60000.0,
            "static_head_m": pump_values[2],
            "shutoff_head_m": pump_values[3],
        },
        "hydraulics": {
            "cv_interstage": [0.0005, 0.0005],
            "gravity_drop_m": [0.30, 0.30],
            "cv_overflow": [0.001, 0.001],
            "overflow_head_floor_m": 1e-9,
        },
        "operation": {
            "circulation_flow_m3s": circulation_lmin / 60000.0,
            "target_levels_m": [value / 1000.0 for value in target_levels_mm],
            "target_temperatures_degC": target_temperatures,
            "initial_levels_m": [value / 1000.0 for value in target_levels_mm],
            "initial_temperatures_degC": initial_temperatures,
            "ambient_temperature_degC": 20.0,
            "control_dt_s": 1.0,
            "duration_s": max(3000.0, maximum_heatup_minutes * 72.0),
        },
        "requirements": {
            "maximum_heatup_time_s": maximum_heatup_minutes * 60.0,
            "temperature_tolerance_degC": 0.75,
            "level_tolerance_m": 0.015,
            "maximum_overshoot_degC": 2.0,
            "maximum_energy_kwh": maximum_energy_kwh,
            "minimum_actuator_margin": 0.04,
            "temperature_trip_degC": 80.0,
            "temperature_hard_limit_degC": 100.0,
            "robustness_pass_rate": robustness_pass_pct / 100.0,
        },
        "uncertainties": {
            "samples": robustness_samples,
            "seed": 20260806,
            "pump_flow_factor": [0.8, 1.0],
            "heater_efficiency_factor": [0.8, 1.0],
            "heat_loss_factor": [1.0, 1.5],
        },
        "references": [
            {
                "id": "interactive-user-input",
                "type": "user-supplied-design",
                "note": "Replace wizard defaults with datasheet or measured values before engineering claims.",
            }
        ],
    }
    if advanced:
        _prompt_advanced(declaration, ask, tell)
    resolved = validate_design_spec(declaration)
    tell("Design inputs validated. SI-unit DesignSpec is ready.")
    return resolved


def _prompt_advanced(declaration, ask, tell):
    tanks = declaration["tanks"]
    spans_mm = _prompt(
        ask,
        tell,
        "Level transmitter spans Tank1,Tank2,Tank3 (mm)",
        [tank["level_sensor_range_m"] * 1000.0 for tank in tanks],
        lambda raw: _csv_numbers(raw, 3, positive=True),
        _format_csv,
    )
    ua_values = _prompt(
        ask,
        tell,
        "Tank heat-loss UA values (W/K)",
        [tank["ua_w_per_k"] for tank in tanks],
        lambda raw: _csv_numbers(raw, 3, nonnegative=True),
        _format_csv,
    )
    low_mm = _prompt(
        ask,
        tell,
        "Low-level trips Tank1,Tank2,Tank3 (mm)",
        [tank["low_level_trip_m"] * 1000.0 for tank in tanks],
        lambda raw: _csv_numbers(raw, 3, nonnegative=True),
        _format_csv,
    )
    high_mm = _prompt(
        ask,
        tell,
        "High-level trips Tank1,Tank2,Tank3 (mm)",
        [tank["high_level_trip_m"] * 1000.0 for tank in tanks],
        lambda raw: _csv_numbers(raw, 3, positive=True),
        _format_csv,
    )
    overflow_mm = _prompt(
        ask,
        tell,
        "Passive overflow levels Tank1,Tank2 (mm)",
        [tanks[index]["overflow_level_m"] * 1000.0 for index in range(2)],
        lambda raw: _csv_numbers(raw, 2, positive=True),
        _format_csv,
    )
    for index, tank in enumerate(tanks):
        tank["level_sensor_range_m"] = spans_mm[index] / 1000.0
        tank["ua_w_per_k"] = ua_values[index]
        tank["low_level_trip_m"] = low_mm[index] / 1000.0
        tank["high_level_trip_m"] = high_mm[index] / 1000.0
        if index < 2:
            tank["overflow_level_m"] = overflow_mm[index] / 1000.0

    hydraulics = declaration["hydraulics"]
    hydraulics["cv_interstage"] = _prompt(
        ask,
        tell,
        "V12,V23 flow coefficients (m^2.5/s)",
        hydraulics["cv_interstage"],
        lambda raw: _csv_numbers(raw, 2, positive=True),
        _format_csv,
    )
    hydraulics["gravity_drop_m"] = _prompt(
        ask,
        tell,
        "V12,V23 gravity drops (m)",
        hydraulics["gravity_drop_m"],
        lambda raw: _csv_numbers(raw, 2, positive=True),
        _format_csv,
    )
    hydraulics["cv_overflow"] = _prompt(
        ask,
        tell,
        "Tank1,Tank2 overflow coefficients (m^2.5/s)",
        hydraulics["cv_overflow"],
        lambda raw: _csv_numbers(raw, 2, positive=True),
        _format_csv,
    )
    operation = declaration["operation"]
    initial_levels_mm = _prompt(
        ask,
        tell,
        "Initial levels Tank1,Tank2,Tank3 (mm)",
        [value * 1000.0 for value in operation["initial_levels_m"]],
        lambda raw: _csv_numbers(raw, 3, positive=True),
        _format_csv,
    )
    operation["initial_levels_m"] = [value / 1000.0 for value in initial_levels_mm]
    operation["ambient_temperature_degC"] = _prompt_number(
        ask, tell, "Ambient temperature (degC)", operation["ambient_temperature_degC"]
    )
    operation["control_dt_s"] = _prompt_number(
        ask, tell, "Control interval (s)", operation["control_dt_s"], positive=True
    )
    operation["duration_s"] = _prompt_number(
        ask, tell, "Simulation duration (s)", operation["duration_s"], positive=True
    )

    requirements = declaration["requirements"]
    requirements["temperature_tolerance_degC"] = _prompt_number(
        ask,
        tell,
        "Final temperature tolerance (degC)",
        requirements["temperature_tolerance_degC"],
        positive=True,
    )
    level_tolerance_mm = _prompt_number(
        ask,
        tell,
        "Final level tolerance (mm)",
        requirements["level_tolerance_m"] * 1000.0,
        positive=True,
    )
    requirements["level_tolerance_m"] = level_tolerance_mm / 1000.0
    requirements["maximum_overshoot_degC"] = _prompt_number(
        ask,
        tell,
        "Maximum temperature overshoot (degC)",
        requirements["maximum_overshoot_degC"],
        nonnegative=True,
    )
    margin_pct = _prompt_number(
        ask,
        tell,
        "Minimum steady actuator margin (percent)",
        requirements["minimum_actuator_margin"] * 100.0,
        lower=0.0,
        upper=100.0,
    )
    requirements["minimum_actuator_margin"] = margin_pct / 100.0
    requirements["temperature_trip_degC"] = _prompt_number(
        ask, tell, "Heater trip temperature (degC)", requirements["temperature_trip_degC"]
    )
    requirements["temperature_hard_limit_degC"] = _prompt_number(
        ask,
        tell,
        "Simulation hard temperature limit (degC)",
        requirements["temperature_hard_limit_degC"],
    )
    uncertainty = declaration["uncertainties"]
    uncertainty["seed"] = _prompt_integer(
        ask, tell, "Robustness seed", uncertainty["seed"], minimum=0
    )
    uncertainty["pump_flow_factor"] = _prompt(
        ask,
        tell,
        "Pump capacity factor min,max",
        uncertainty["pump_flow_factor"],
        lambda raw: _factor_range(raw),
        _format_csv,
    )
    uncertainty["heater_efficiency_factor"] = _prompt(
        ask,
        tell,
        "Heater efficiency factor min,max",
        uncertainty["heater_efficiency_factor"],
        lambda raw: _factor_range(raw, upper=1.0),
        _format_csv,
    )
    uncertainty["heat_loss_factor"] = _prompt(
        ask,
        tell,
        "Heat-loss factor min,max",
        uncertainty["heat_loss_factor"],
        lambda raw: _factor_range(raw),
        _format_csv,
    )


def _prompt(ask, tell, label, default, parser, formatter):
    while True:
        raw = ask(f"{label} [{formatter(default)}]: ").strip()
        if not raw:
            return copy.deepcopy(default)
        try:
            return parser(raw)
        except (TypeError, ValueError) as exc:
            tell(f"Invalid value: {exc}")


def _prompt_number(
    ask,
    tell,
    label,
    default,
    *,
    positive=False,
    nonnegative=False,
    lower=None,
    upper=None,
):
    def parse(raw):
        value = float(raw)
        if positive and value <= 0.0:
            raise ValueError("must be positive")
        if nonnegative and value < 0.0:
            raise ValueError("must be non-negative")
        if lower is not None and value < lower:
            raise ValueError(f"must be >= {lower}")
        if upper is not None and value > upper:
            raise ValueError(f"must be <= {upper}")
        return value

    return _prompt(ask, tell, label, float(default), parse, lambda value: f"{value:g}")


def _prompt_integer(ask, tell, label, default, *, minimum=None):
    def parse(raw):
        value = int(raw)
        if minimum is not None and value < minimum:
            raise ValueError(f"must be >= {minimum}")
        return value

    return _prompt(ask, tell, label, int(default), parse, str)


def _csv_numbers(raw, length, *, positive=False, nonnegative=False):
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != length:
        raise ValueError(f"expected {length} comma-separated values")
    values = [float(part) for part in parts]
    if positive and any(value <= 0.0 for value in values):
        raise ValueError("all values must be positive")
    if nonnegative and any(value < 0.0 for value in values):
        raise ValueError("all values must be non-negative")
    return values


def _parse_heaters(raw):
    if raw.strip().lower() in {"none", "zero", "0"}:
        return []
    heaters = []
    seen = set()
    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError("use tank:power_kW:efficiency_pct for each heater")
        tank = int(parts[0])
        power_kw = float(parts[1])
        efficiency_pct = float(parts[2])
        if tank not in (1, 2, 3) or tank in seen:
            raise ValueError("heater tank IDs must be unique values from 1 to 3")
        if power_kw <= 0.0:
            raise ValueError("heater power must be positive")
        if not 0.0 < efficiency_pct <= 100.0:
            raise ValueError("heater efficiency must be in (0, 100]")
        seen.add(tank)
        heaters.append(
            {
                "tank": tank,
                "power_w": power_kw * 1000.0,
                "efficiency": efficiency_pct / 100.0,
            }
        )
    return heaters


def _factor_range(raw, *, upper=None):
    values = _csv_numbers(raw, 2, positive=True)
    if values[0] > values[1]:
        raise ValueError("minimum must not exceed maximum")
    if upper is not None and values[1] > upper:
        raise ValueError(f"maximum must be <= {upper}")
    return values


def _format_csv(values):
    return ",".join(f"{float(value):g}" for value in values)


def _format_heaters(heaters):
    if not heaters:
        return "none"
    return ",".join(
        f"{heater['tank']}:{heater['power_w'] / 1000.0:g}:"
        f"{heater['efficiency'] * 100.0:g}"
        for heater in heaters
    )


__all__ = ["prompt_design_spec"]

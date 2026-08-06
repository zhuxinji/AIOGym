"""Generic structural and numerical readiness checks for process models."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .integration import Integrator
from .model import ProcessModelContract


def _validate_model_contract(model):
    if not isinstance(model, ProcessModelContract):
        raise TypeError("model must inherit ProcessModelContract")
    if len(model.initial_state()) != len(model.state_schema()):
        raise TypeError("state schema does not match initial state")
    if len(model.action_schema()) != model.action_dim():
        raise TypeError("action schema does not match action dimension")
    return model


def _finite_vector(values) -> bool:
    return isinstance(values, Sequence) and all(math.isfinite(float(value)) for value in values)


def validate_model_readiness(model_or_name, *, integration_dt: float | None = None) -> dict:
    """Run model-agnostic contract, derivative, output, and integration checks.

    This intentionally does not claim mass/energy conservation or parameter
    fidelity; those checks require scenario-specific equations and references.
    """

    if isinstance(model_or_name, str):
        raise TypeError("readiness validation requires a resolved model instance")
    model = model_or_name
    checks = []

    def record(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    try:
        _validate_model_contract(model)
        record("model_contract", True)
    except Exception as exc:  # report all checks through one stable surface
        record("model_contract", False, exc)

    x0 = list(model.initial_state())
    u0 = model.action_vector(model.default_action())
    env = model.runtime_env(model.disturbance_defaults())
    try:
        dx = list(model.dynamics(x0, u0, env))
        record("finite_derivatives", len(dx) == len(x0) and _finite_vector(dx), f"dimension={len(dx)}")
    except Exception as exc:
        record("finite_derivatives", False, exc)
    try:
        outputs = model.outputs(x0)
        y = outputs.get("y", ()) if isinstance(outputs, Mapping) else ()
        record("finite_outputs", _finite_vector(y), f"dimension={len(y)}")
    except Exception as exc:
        record("finite_outputs", False, exc)
    try:
        settings = model.solver_settings()
        method = settings.get("method")
        max_step = float(settings.get("max_step"))
        record("solver_configuration", method == "rk4" and math.isfinite(max_step) and max_step > 0,
               f"method={method}, max_step={max_step}")
    except Exception as exc:
        record("solver_configuration", False, exc)
        max_step = float(getattr(model, "dt_micro", 0.02) or 0.02)
    try:
        dt = float(integration_dt if integration_dt is not None else max_step)
        integrator = Integrator(model)
        x1 = integrator.step(dt, u0, env)
        record("finite_integration_step", len(x1) == len(x0) and _finite_vector(x1), f"dt={dt}")
    except Exception as exc:
        record("finite_integration_step", False, exc)

    scenario_validator = getattr(model, "physical_validation_checks", None)
    if callable(scenario_validator):
        try:
            for row in scenario_validator():
                record(row["name"], row["passed"], row.get("detail", ""))
        except Exception as exc:
            record("scenario_physical_validation", False, exc)

    checked_names = {row["name"] for row in checks if row["passed"]}
    not_checked = ["mass_balance", "energy_balance", "reference_parameter_fidelity"]
    if "mass_balance" in checked_names:
        not_checked.remove("mass_balance")
    if "energy_balance" in checked_names:
        not_checked.remove("energy_balance")
    if "reference_parameters" in checked_names:
        not_checked.remove("reference_parameter_fidelity")
    not_applicable = list(getattr(model, "not_applicable_physics", ()))
    not_checked = [name for name in not_checked if name not in not_applicable]

    return {
        "scenario": model.scenario,
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "not_checked": not_checked,
        "not_applicable": not_applicable,
    }

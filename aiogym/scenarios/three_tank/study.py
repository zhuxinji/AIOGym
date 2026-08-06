"""Scenario-owned engineering checks for parameterized three-tank plants."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aiogym.core import CheckResult, ResolvedPlant

from .model import ThreeTankModel, _legacy_design_spec


class ThreeTankStudyProvider:
    categories = ("static", "model", "steady_state", "safety", "dynamic", "robustness")

    def checks(self, plant: ResolvedPlant):
        model = ThreeTankModel(plant)
        spec = _legacy_design_spec(plant)
        from .checks import (
            _safety_interlock_assessment,
            _static_assessment,
            _steady_state_assessment,
        )
        from aiogym.core.validation import validate_model_readiness

        static = _static_assessment(model._model, spec)
        readiness = validate_model_readiness(model._model)
        steady = _steady_state_assessment(model._model, spec)
        safety = _safety_interlock_assessment(model._model, spec)
        return (
            _result(
                "static_engineering",
                "static",
                static["passed"],
                static,
                static["failed_checks"],
            ),
            _result(
                "model_readiness",
                "model",
                readiness["passed"],
                readiness,
                [row["name"] for row in readiness["checks"] if not row["passed"]],
            ),
            _result(
                "steady_state_feasibility",
                "steady_state",
                steady["passed"],
                steady,
                steady["reasons"],
            ),
            _result(
                "hardware_interlocks",
                "safety",
                safety["passed"],
                safety,
                safety["failed_checks"],
            ),
        )

    def dynamic_requirements(self, plant: ResolvedPlant) -> Mapping:
        return {
            "operation": dict(plant.parameters["operation"]),
            "requirements": dict(plant.parameters["requirements"]),
        }

    def steady_check(self, plant: ResolvedPlant, disturbances: Mapping):
        model = ThreeTankModel(plant)
        spec = _legacy_design_spec(plant)
        from .checks import _steady_state_assessment

        metrics = _steady_state_assessment(model._model, spec, env=disturbances)
        return _result(
            "steady_state_feasibility",
            "steady_state",
            metrics["passed"],
            metrics,
            metrics["reasons"],
        )

    def default_robustness(self, plant: ResolvedPlant) -> tuple[int, int]:
        uncertainty = dict(plant.parameters.get("uncertainties", {}))
        return int(uncertainty.get("samples", 0)), int(uncertainty.get("seed", 0))

    def sample_disturbances(
        self, plant: ResolvedPlant, rng: np.random.Generator
    ) -> dict[str, float]:
        uncertainty = dict(plant.parameters.get("uncertainties", {}))
        operation = dict(plant.parameters["operation"])

        def sample(name, default):
            bounds = uncertainty.get(name, (default, default))
            return float(rng.uniform(float(bounds[0]), float(bounds[1])))

        return {
            "t_amb": float(operation["ambient_temperature_degC"]),
            "pump_flow_factor": sample("pump_flow_factor", 1.0),
            "heater_efficiency": sample("heater_efficiency_factor", 1.0),
            "heat_loss_factor": sample("heat_loss_factor", 1.0),
        }


def _result(name, category, passed, metrics, reasons):
    reason_list = [str(reason) for reason in reasons]
    summary = "passed" if passed else "; ".join(reason_list) or "failed"
    return CheckResult(
        name=name,
        category=category,
        passed=bool(passed),
        summary=summary,
        metrics=metrics,
    )


__all__ = ["ThreeTankStudyProvider"]

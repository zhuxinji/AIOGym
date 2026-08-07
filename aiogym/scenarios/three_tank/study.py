"""Scenario-owned engineering checks for parameterized three-tank plants."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aiogym.core import CheckResult, ResolvedPlant

from .model import ThreeTankModel


class ThreeTankStudyProvider:
    categories = ("static", "model", "steady_state", "safety", "dynamic", "robustness")

    def checks(self, plant: ResolvedPlant):
        model = ThreeTankModel(plant)
        if "parameters" in plant.config.plant:
            from aiogym.core.validation import validate_model_readiness

            readiness = validate_model_readiness(model._model)
            return (
                _result(
                    "model_readiness",
                    "model",
                    readiness["passed"],
                    readiness,
                    [
                        row["name"]
                        for row in readiness["checks"]
                        if not row["passed"]
                    ],
                ),
            )
        context = model._model.study_context
        from .checks import (
            _safety_interlock_assessment,
            _static_assessment,
            _steady_state_assessment,
        )
        from aiogym.core.validation import validate_model_readiness

        static = _static_assessment(model._model, context)
        readiness = validate_model_readiness(model._model)
        steady = _steady_state_assessment(model._model, context)
        safety = _safety_interlock_assessment(model._model, context)
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
        if "parameters" in plant.config.plant:
            condition = plant.config.conditions[plant.config.default_condition]
            return {
                "operation": {
                    "target_levels_m": list(condition.reference[:3]),
                    "target_temperatures_degC": list(condition.reference[3:]),
                },
                "requirements": {
                    "level_tolerance_m": 0.05,
                    "temperature_tolerance_degC": 5.0,
                    "maximum_heatup_time_s": condition.horizon
                    * condition.control_dt,
                    "maximum_overshoot_degC": 100.0,
                    "maximum_energy_kwh": 1.0e9,
                },
            }
        context = ThreeTankModel(plant)._model.study_context
        return {
            "operation": dict(context["operation"]),
            "requirements": dict(context["requirements"]),
        }

    def steady_check(self, plant: ResolvedPlant, disturbances: Mapping):
        model = ThreeTankModel(plant)
        if "parameters" in plant.config.plant:
            values = model._model.tracking_steady_state_action(
                plant.config.conditions[
                    plant.config.default_condition
                ].reference
            )
            return _result(
                "steady_state_feasibility",
                "steady_state",
                values is not None,
                {"action": values},
                [] if values is not None else ["steady-state action unavailable"],
            )
        context = model._model.study_context
        from .checks import _steady_state_assessment

        metrics = _steady_state_assessment(model._model, context, env=disturbances)
        return _result(
            "steady_state_feasibility",
            "steady_state",
            metrics["passed"],
            metrics,
            metrics["reasons"],
        )

    def default_robustness(self, plant: ResolvedPlant) -> tuple[int, int]:
        uncertainty = dict(plant.config.study.get("uncertainty", {}))
        return int(uncertainty.get("samples", 0)), int(uncertainty.get("seed", 0))

    def sample_disturbances(
        self, plant: ResolvedPlant, rng: np.random.Generator
    ) -> dict[str, float]:
        if "parameters" in plant.config.plant:
            defaults = ThreeTankModel(plant).default_disturbances()
            return dict(defaults)
        operation = ThreeTankModel(plant)._model.operation
        uncertainty = dict(plant.config.study.get("uncertainty", {}))

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

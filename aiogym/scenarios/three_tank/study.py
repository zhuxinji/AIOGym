"""Scenario-owned engineering checks for parameterized three-tank plants."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aiogym.core import CheckResult, StudyContext

from .equipment import ThreeTankDesignModel


class ThreeTankStudyProvider:
    categories = ("static", "model", "steady_state", "safety", "dynamic", "robustness")

    def checks(self, context: StudyContext):
        model = context.model._model
        from aiogym.core.validation import validate_model_readiness

        readiness = validate_model_readiness(model)
        results = []
        if isinstance(model, ThreeTankDesignModel):
            from .checks import (
                _safety_interlock_assessment,
                _static_assessment,
                _steady_state_assessment,
            )

            inputs = model.study_context(context.condition)
            static = _static_assessment(model, inputs)
            steady = _steady_state_assessment(
                model,
                inputs,
                reference=context.condition.reference,
                env=context.condition.disturbances,
            )
            safety = _safety_interlock_assessment(model, inputs)
            results.extend(
                (
                    _result(
                        "static_engineering",
                        "static",
                        static["passed"],
                        static,
                        static["failed_checks"],
                    ),
                    _readiness_result(readiness),
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
            )
        else:
            results.append(_readiness_result(readiness))

        requirements = self.dynamic_requirements(context)["requirements"]
        if not requirements["assessment_horizon_sufficient"]:
            results.append(
                _result(
                    "insufficient_assessment_horizon",
                    "study_configuration",
                    False,
                    {
                        "available_duration_s": requirements["available_duration_s"],
                        "maximum_heatup_time_s": requirements[
                            "maximum_heatup_time_s"
                        ],
                    },
                    [
                        "available duration is shorter than maximum heat-up assessment time"
                    ],
                )
            )
        return tuple(results)

    def dynamic_requirements(self, context: StudyContext) -> Mapping:
        model = context.model._model
        if isinstance(model, ThreeTankDesignModel):
            inputs = model.study_context(context.condition)
            return {
                "operation": dict(inputs["operation"]),
                "requirements": dict(inputs["requirements"]),
            }

        condition = context.condition
        available_duration = float(condition.control_dt * condition.horizon)
        configured = dict(context.plant.config.study.get("requirements", {}))
        maximum_heatup = float(
            configured.get("maximum_heatup_time_s", available_duration)
        )
        return {
            "operation": {
                "target_levels_m": list(condition.reference[:3]),
                "target_temperatures_degC": list(condition.reference[3:]),
                "initial_levels_m": list(condition.initial_state[0::2]),
                "initial_temperatures_degC": list(condition.initial_state[1::2]),
                "ambient_temperature_degC": float(
                    condition.disturbances.get("t_amb", 20.0)
                ),
                "control_dt_s": float(condition.control_dt),
                "duration_s": available_duration,
            },
            "requirements": {
                "level_tolerance_m": float(
                    configured.get("level_tolerance_m", 0.05)
                ),
                "temperature_tolerance_degC": float(
                    configured.get("temperature_tolerance_degC", 5.0)
                ),
                "maximum_heatup_time_s": maximum_heatup,
                "maximum_overshoot_degC": float(
                    configured.get("maximum_overshoot_degC", 100.0)
                ),
                "maximum_energy_kwh": float(
                    configured.get("maximum_energy_kwh", 1.0e9)
                ),
                "available_duration_s": available_duration,
                "assessment_horizon_sufficient": maximum_heatup
                <= available_duration,
            },
        }

    def steady_check(
        self,
        context: StudyContext,
        disturbances: Mapping[str, float],
    ):
        model = context.model._model
        if not isinstance(model, ThreeTankDesignModel):
            values = model.tracking_steady_state_action(context.condition.reference)
            return _result(
                "steady_state_feasibility",
                "steady_state",
                values is not None,
                {
                    "action": values,
                    "reference": list(context.condition.reference),
                    "condition_hash": context.condition.condition_hash,
                },
                [] if values is not None else ["steady-state action unavailable"],
            )

        from .checks import _steady_state_assessment

        inputs = model.study_context(context.condition)
        metrics = _steady_state_assessment(
            model,
            inputs,
            reference=context.condition.reference,
            env=disturbances,
        )
        return _result(
            "steady_state_feasibility",
            "steady_state",
            metrics["passed"],
            metrics,
            metrics["reasons"],
        )

    def default_robustness(self, context: StudyContext) -> tuple[int, int]:
        uncertainty = dict(context.plant.config.study.get("uncertainty", {}))
        return int(uncertainty.get("samples", 0)), int(uncertainty.get("seed", 0))

    def sample_disturbances(
        self, context: StudyContext, rng: np.random.Generator
    ) -> dict[str, float]:
        defaults = dict(context.model.default_disturbances())
        defaults.update(context.condition.disturbances)
        model = context.model._model
        if not isinstance(model, ThreeTankDesignModel):
            return defaults
        uncertainty = dict(context.plant.config.study.get("uncertainty", {}))

        def sample(name, default):
            bounds = uncertainty.get(name, (default, default))
            return float(rng.uniform(float(bounds[0]), float(bounds[1])))

        return {
            "t_amb": float(defaults["t_amb"]),
            "pump_flow_factor": sample(
                "pump_flow_factor", defaults["pump_flow_factor"]
            ),
            "heater_efficiency": sample(
                "heater_efficiency_factor", defaults["heater_efficiency"]
            ),
            "heat_loss_factor": sample(
                "heat_loss_factor", defaults["heat_loss_factor"]
            ),
        }


def _readiness_result(readiness):
    return _result(
        "model_readiness",
        "model",
        readiness["passed"],
        readiness,
        [row["name"] for row in readiness["checks"] if not row["passed"]],
    )


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

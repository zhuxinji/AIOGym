"""Read-only environment information shared by Python queries and the CLI."""

from __future__ import annotations

import math

from .contracts import environment_interface
from .io import jsonable


def parameter_information(model):
    values = model.resolved_parameters
    units = model.parameter_units
    missing, unknown = sorted(set(values) - set(units)), sorted(set(units) - set(values))
    if missing or unknown:
        raise ValueError(
            f"parameter unit metadata mismatch for scenario {model.scenario!r}; "
            f"missing: {missing}; unknown: {unknown}"
        )
    if any(not isinstance(unit, str) or not unit for unit in units.values()):
        raise ValueError(f"parameter units must be non-empty strings for {model.scenario!r}")
    descriptions = getattr(model, "parameter_metadata", {})
    defaults = getattr(model, "_parameter_defaults", {})
    return jsonable({
        name: {
            "value": values[name],
            "default": defaults.get(name),
            "unit": units[name],
            "description": descriptions.get(name, ("Not provided", "Not provided"))[0],
            "allowed_values": descriptions.get(name, ("Not provided", "Not provided"))[1],
        }
        for name in sorted(values)
    })


def variable_information(model, category, rows=None):
    if rows is None:
        rows = getattr(model, f"{category}_schema")()
    descriptions = getattr(model, "variable_descriptions", {})
    metadata_method = getattr(model, f"{category}_metadata", None)
    metadata = {} if metadata_method is None else metadata_method()
    slew = {}
    if category == "action":
        limits = model.action_slew_limits()
        if limits is not None:
            slew = {row["name"]: limit for row, limit in zip(model.action_schema(), limits)}
    result = {}
    for index, source in enumerate(rows):
        row = dict(source)
        name = row.pop("name")
        row.update(index=index, description=descriptions.get(name, "Not provided"))
        row.update(metadata.get(name, {}))
        if name in slew:
            row["max_change_per_step"] = slew[name]
        # Preserve the distinction between an absent bound and an unbounded side.
        for bound in ("low", "high"):
            if bound in row and row[bound] is not None and math.isinf(row[bound]):
                row[bound] = "-Infinity" if row[bound] < 0 else "Infinity"
        result[name] = row
    return jsonable(result)


def state_limit_rules(bounds, units):
    """Describe explicitly selected termination bounds; never infer constraints."""
    return {
        name: {
            "kind": "termination",
            "condition": f"{name} < {low:g} or {name} > {high:g} {units[name]}",
            "effect": "Terminate episode",
        }
        for name, (low, high) in bounds.items()
    }


def reward_information(scenario):
    return {
        name: {
            "success_criterion": reward.success_criterion,
            "primary_metric": reward.primary_metric,
            "metric_direction": reward.metric_direction,
            "safety_violation_penalty": reward.safety_violation_penalty,
        }
        for name, reward in sorted(scenario.rewards.items())
    }


def benchmark_information(scenario):
    return {
        name: {
            "description": benchmark.description,
            "horizon": benchmark.horizon,
            "reward": benchmark.reward_id,
            "parameters": "scenario-defaults",
            "case_selection": "Explicit non-negative seed; reset required",
            "ranking_metrics": [
                {"name": metric, "direction": direction}
                for metric, direction in benchmark.ranking_metrics
            ],
            "success_criterion": scenario.rewards[benchmark.reward_id].success_criterion,
        }
        for name, benchmark in sorted(scenario.benchmarks.items())
    }


class EnvironmentInformation:
    """Descriptions and information properties for environments and their wrappers."""

    @property
    def parameters(self):
        """Independent snapshot of the effective model parameters."""
        return jsonable(self.unwrapped.model.resolved_parameters)

    @property
    def states(self):
        return variable_information(self.unwrapped.model, "state")

    @property
    def actions(self):
        return variable_information(
            self.unwrapped.model, "action", environment_interface(self)["action"]
        )

    @property
    def observations(self):
        return variable_information(
            self.unwrapped.model, "observation", environment_interface(self)["observation"]
        )

    @property
    def outputs(self):
        return variable_information(self.unwrapped.model, "output")

    @property
    def rewards(self):
        return reward_information(self.unwrapped.scenario)

    @property
    def benchmarks(self):
        return benchmark_information(self.unwrapped.scenario)

    @property
    def safety_rules(self):
        method = getattr(self.unwrapped.model, "safety_metadata", None)
        return None if method is None else jsonable(method())

    def describe(self):
        """Return an independent description without resetting, stepping or sampling."""
        base = self.unwrapped
        model = base.model
        episode = base.episode.as_dict()
        episode.update(
            status="resolved" if base.episode_parameters else "template",
            steps_completed=base._step_index,
            duration=base.episode.horizon * base.control_dt,
            selection=dict(base.episode_parameters),
        )
        if base.episode_parameters:
            episode["current_reference"] = base.y_sp
            episode["current_disturbances"] = base.disturbances
        disturbance_defaults = model.default_disturbances()
        disturbances = {
            row["name"]: {**row, "default": disturbance_defaults[row["name"]]}
            for row in getattr(model, "input_disturbances", ())
        }
        return jsonable({
            "environment": {
                "scenario": base.scenario.id,
                "description": type(model).__doc__ or "Not provided",
                "reward": base.reward.id,
                "benchmark": None if base.benchmark is None else base.benchmark.id,
                "control_dt": base.control_dt,
                "time_unit": getattr(model, "time_unit", "s"),
                "integration_dt_max": model.dt_micro,
                "integration_method": "RK4",
                "adapter": environment_interface(self).get("adapter"),
            },
            "parameters": parameter_information(model),
            "states": self.states,
            "actions": self.actions,
            "observations": self.observations,
            "outputs": self.outputs,
            "rewards": self.rewards,
            "benchmarks": self.benchmarks,
            "safety_rules": self.safety_rules,
            "disturbances": disturbances,
            "episode": episode,
            "configuration": base.runtime_config,
        })

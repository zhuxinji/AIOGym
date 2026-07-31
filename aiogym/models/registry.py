import copy
import math
from numbers import Real
from typing import Mapping

from .._internal.identifiers import (
    internal_scenario_id,
    require_canonical_scenario_id,
    scenario_catalog_text,
)
from .core import ProcessModelContract, _is_model_instance
from .scenarios import (
    CascadeModel,
    RecirculatingCascadeModel,
    QuadrupleModel,
    CSTRModel,
    HVACModel,
    ExtractionModel,
    FiredHeaterModel,
    CrystallizationModel,
)


MODELS = {
    "cascade": CascadeModel,
    "cascade_recirculating": RecirculatingCascadeModel,
    "quadruple": QuadrupleModel,
    "cstr": CSTRModel,
    "hvac": HVACModel,
    "extraction": ExtractionModel,
    "heater": FiredHeaterModel,
    "crystallization": CrystallizationModel,
}
BUILTIN_MODELS = dict(MODELS)


def validate_model_contract(model):
    required_methods = (
        "action_dim", "initial_state", "action_vector",
        "dynamics", "outputs", "measurement", "controlled_output", "setpoint_vector",
    )
    missing = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"custom process model is missing required methods: {', '.join(missing)}")
    schema_methods = ("state_schema", "action_schema", "disturbance_schema", "disturbance_defaults")
    missing_schema = [name for name in schema_methods if not callable(getattr(model, name, None))]
    if missing_schema:
        raise TypeError(
            "custom process model should inherit ProcessModelContract or implement schema methods: "
            + ", ".join(missing_schema)
        )
    if not hasattr(model, "p") or not isinstance(model.p, dict):
        raise TypeError("custom process model must expose a parameter dict as .p")
    if not hasattr(model, "scenario") or not getattr(model, "scenario"):
        raise TypeError("custom process model must expose a non-empty .scenario")
    if not hasattr(model, "n") or int(model.n) <= 0:
        raise TypeError("custom process model must expose a positive .n")
    has_dynamics_implementation = (
        callable(getattr(model, "_dynamics", None))
        or type(model).dynamics is not ProcessModelContract.dynamics
    )
    if not has_dynamics_implementation:
        raise TypeError(
            "custom process model must implement _dynamics(x, u, d, ops) or override dynamics()"
        )
    if len(model.initial_state()) != len(model.state_schema()):
        raise TypeError("custom process model state_schema() length must match initial_state()")
    if len(model.action_schema()) != model.action_dim():
        raise TypeError("custom process model action_schema() length must match action_dim()")
    return model


def register_model(name, model_factory, *, replace=False):
    if not isinstance(name, str) or not name:
        raise ValueError("model name must be a non-empty string")
    reserved = internal_scenario_id(name)
    if reserved != name and reserved in MODELS:
        raise ValueError(
            f"model name {name!r} is the canonical alias of registered scenario "
            f"{reserved!r}; register custom models with a distinct ID"
        )
    if name in MODELS and not replace:
        raise ValueError(f"model '{name}' is already registered")
    if _is_model_instance(model_factory):
        sample = validate_model_contract(model_factory)
        _validate_registered_name(name, sample)
        MODELS[name] = lambda: copy.deepcopy(model_factory)
    elif isinstance(model_factory, type) or callable(model_factory):
        sample = validate_model_contract(model_factory())
        _validate_registered_name(name, sample)
        MODELS[name] = model_factory
    else:
        raise TypeError("model_factory must be a model instance, class, or zero-argument factory")


def _validate_registered_name(name, model):
    if model.scenario != name:
        raise ValueError(
            f"registered model name {name!r} must match model.scenario {model.scenario!r}"
        )


def apply_model_params(model, params: Mapping | None = None):
    """Apply parameter overrides to a process model instance."""

    if not params:
        return model
    if not isinstance(params, Mapping):
        raise TypeError("model_params must be a mapping of parameter names to values")
    unknown = [name for name in params if name not in model.p]
    if unknown:
        known = ", ".join(sorted(model.p))
        raise KeyError(
            f"unknown model parameter(s) for {model.scenario}: {', '.join(unknown)}. "
            f"Known parameters: {known}"
        )
    validated = {
        name: _validated_parameter_value(model, name, value)
        for name, value in params.items()
    }
    for name, value in validated.items():
        model.p[name] = value
    return model


def _validated_parameter_value(model, name, value):
    current = model.p[name]
    bounds = getattr(model, "param_bounds", {}).get(name)
    current_values = _parameter_sequence(current)
    override_values = _parameter_sequence(value)
    if current_values is not None:
        if override_values is None:
            raise TypeError(f"model parameter '{name}' must contain {len(current_values)} numeric values")
        if len(override_values) != len(current_values):
            raise ValueError(
                f"model parameter '{name}' must contain {len(current_values)} values, "
                f"got {len(override_values)}"
            )
        return [
            _validated_scalar_parameter(name, item, bounds, index=index)
            for index, item in enumerate(override_values)
        ]
    if override_values is not None:
        raise TypeError(f"model parameter '{name}' must be a numeric scalar")
    return _validated_scalar_parameter(name, value, bounds)


def _parameter_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    return None


def _validated_scalar_parameter(name, value, bounds, index=None):
    suffix = f"[{index}]" if index is not None else ""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"model parameter '{name}{suffix}' must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"model parameter '{name}{suffix}' must be finite")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        lo, hi = bounds
        if lo is not None and number < float(lo):
            raise ValueError(f"model parameter '{name}{suffix}' must be >= {lo}")
        if hi is not None and number > float(hi):
            raise ValueError(f"model parameter '{name}{suffix}' must be <= {hi}")
    return copy.deepcopy(value)


def unregister_model(name):
    resolved = internal_scenario_id(name)
    if resolved in BUILTIN_MODELS:
        raise ValueError(f"built-in model '{name}' cannot be unregistered")
    MODELS.pop(resolved, None)


def make_model(scenario="cascade"):
    if _is_model_instance(scenario):
        return validate_model_contract(copy.deepcopy(scenario))
    if isinstance(scenario, type) or (callable(scenario) and not isinstance(scenario, str)):
        return validate_model_contract(scenario())
    requested = scenario
    require_canonical_scenario_id(requested)
    scenario = internal_scenario_id(scenario)
    if scenario not in MODELS:
        available = scenario_catalog_text(tuple(MODELS))
        raise ValueError(
            f"unknown scenario ID {requested!r}; available scenario IDs: {available}"
        )
    factory = MODELS[scenario]
    model = factory if _is_model_instance(factory) else factory()
    return validate_model_contract(model)

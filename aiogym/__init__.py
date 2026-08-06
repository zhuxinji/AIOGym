"""Small public API for AIO-Gym 0.2."""
from __future__ import annotations

import warnings


__version__ = "0.3.0"


def _register():
    import aiogym.scenarios  # noqa: F401


def list_scenarios():
    _register()
    from .core import list_scenarios as implementation

    return implementation()


def list_tasks(scenario: str | None = None):
    _register()
    from .core import list_tasks as implementation

    return implementation(scenario=scenario)


def list_plants(scenario: str):
    _register()
    from .core import list_plants as implementation

    return implementation(scenario=scenario)


def list_conditions(scenario: str, plant=None):
    _register()
    from .core import resolve_plant

    resolved = resolve_plant(scenario, plant)
    return tuple(sorted(resolved.config.conditions))


def make_env(task: str, *, plant=None, condition=None, preset=None, **legacy):
    """Create an environment from a Task and optional PlantConfig.

    A narrow compatibility path accepts the old ``scenario + case/reward_spec``
    call shape for one release. It deliberately does not recreate Track logic.
    """
    _register()
    if preset is not None:
        if condition is not None:
            raise TypeError("condition and deprecated preset cannot both be provided")
        warnings.warn(
            "preset= is deprecated; use condition=",
            DeprecationWarning,
            stacklevel=2,
        )
        condition = preset
    if "/" not in task or legacy:
        warnings.warn(
            "make_env(scenario, case=..., reward_spec=...) is deprecated; use "
            "make_env('scenario/objective', plant=..., condition=...)",
            DeprecationWarning,
            stacklevel=2,
        )
        unknown = set(legacy) - {"case", "reward_spec", "info_level"}
        if unknown:
            raise TypeError(f"unsupported legacy make_env arguments: {sorted(unknown)}")
        reward = str(legacy.get("reward_spec") or "regulation").lower()
        objective = "economic" if "economic" in reward else "regulation"
        task = f"{task}/{objective}"
        if condition is None and legacy.get("case") is not None:
            condition = str(legacy["case"])
    from .core import make_env as implementation

    return implementation(task, plant=plant, condition=condition)


def load_plant(source):
    from .workflows import load_plant as implementation

    return implementation(source)


def make_controller(controller_id: str, *, env, profile=None, config=None):
    from .controllers.base import make_controller as implementation

    return implementation(controller_id, env=env, profile=profile, config=config)


def study(*args, **kwargs):
    from .workflows import study as implementation

    return implementation(*args, **kwargs)


def collect(**kwargs):
    from .workflows import collect as implementation

    return implementation(**kwargs)


def train(**kwargs):
    from .workflows import train as implementation

    return implementation(**kwargs)


def evaluate(*args, **kwargs):
    from .workflows import evaluate as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "__version__",
    "collect",
    "evaluate",
    "list_scenarios",
    "list_plants",
    "list_conditions",
    "list_tasks",
    "load_plant",
    "make_controller",
    "make_env",
    "study",
    "train",
]

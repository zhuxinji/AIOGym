"""Controller registry and built-in factories."""
from __future__ import annotations

from typing import Callable, Mapping, Any

from .configs import _controller_params, _merged_controller_config
from .contracts import Controller


ControllerFactory = Callable[..., Controller]
_REGISTRY: dict[str, ControllerFactory] = {}
BUILTIN_CONTROLLERS: dict[str, ControllerFactory] = {}


def register_controller(name: str, factory: ControllerFactory, *, replace: bool = False) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("controller name must be a non-empty string")
    if not callable(factory):
        raise TypeError("controller factory must be callable")
    key = name.lower()
    if key in _REGISTRY and not replace:
        raise ValueError(f"controller '{key}' is already registered")
    _REGISTRY[key] = factory


def _controller_ids() -> tuple[str, ...]:
    return tuple(sorted(name for name in _REGISTRY if name != "oracle"))


def unregister_controller(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("controller name must be a non-empty string")
    key = name.lower()
    if key in BUILTIN_CONTROLLERS:
        _REGISTRY[key] = BUILTIN_CONTROLLERS[key]
    else:
        _REGISTRY.pop(key, None)


def make_controller(
    name: str,
    model=None,
    scenario: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> Controller:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown controller ID {name!r}; available controller IDs: "
            f"{', '.join(_controller_ids())}"
        )
    requested_scenario = scenario or dict(config or {}).get("scenario")
    cfg = _merged_controller_config(key, requested_scenario, config)
    if model is None:
        from ..models.registry import make_model

        model = make_model(requested_scenario or cfg.pop("scenario", "cstr"))
    return _REGISTRY[key](
        model=model,
        scenario=requested_scenario or getattr(model, "scenario", None),
        config=cfg,
    )


def _pid_factory(model=None, scenario=None, config=None):
    from .pid import PIDAgent

    cfg = dict(config or {})
    agent = PIDAgent(model, **_controller_params(cfg))
    agent.control_structure = cfg.get("control_structure", "fixed_sp_pid")
    return agent


def _hold_factory(model=None, scenario=None, config=None):
    del scenario, config
    from .hold import HoldController

    return HoldController(model)


def _mpc_factory(model=None, scenario=None, config=None):
    from .mpc import MPCAgent

    cfg = dict(config or {})
    agent = MPCAgent(model, **_controller_params(cfg))
    agent.control_structure = cfg.get("control_structure", "fixed_sp_mpc")
    return agent


def _oracle_factory(model=None, scenario=None, config=None):
    from .oracle import OracleAgent

    cfg = dict(config or {})
    params = _controller_params(cfg)
    for name in ("goal", "reward_spec"):
        if cfg.get(name) is not None:
            params[name] = cfg[name]
    agent = OracleAgent(
        scenario or model.scenario,
        model=model,
        **params,
    )
    agent.control_structure = cfg.get("control_structure", "nmpc_oracle")
    return agent


register_controller("hold", _hold_factory)
register_controller("pid", _pid_factory)
register_controller("mpc", _mpc_factory)
register_controller("oracle", _oracle_factory)
BUILTIN_CONTROLLERS.update(_REGISTRY)

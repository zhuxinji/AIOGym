"""Controller registry and built-in factories."""
from __future__ import annotations

from typing import Callable, Mapping, Any

from .adapters import PolicyController, SB3PolicyController
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
    return tuple(sorted(_REGISTRY))


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
    policy=None,
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
        from ..models import make_model

        model = make_model(requested_scenario or cfg.pop("scenario", "cstr"))
    return _REGISTRY[key](
        model=model,
        scenario=requested_scenario or getattr(model, "scenario", None),
        config=cfg,
        policy=policy,
    )


def _pid_factory(model=None, scenario=None, config=None, policy=None):
    from .pid import PIDAgent

    cfg = dict(config or {})
    agent = PIDAgent(model, **_controller_params(cfg))
    agent.control_structure = cfg.get("control_structure", "fixed_sp_pid")
    return agent


def _hold_factory(model=None, scenario=None, config=None, policy=None):
    del scenario, config, policy
    from .hold import HoldController

    return HoldController(model)


def _mpc_factory(model=None, scenario=None, config=None, policy=None):
    from .mpc import MPCAgent

    cfg = dict(config or {})
    agent = MPCAgent(model, **_controller_params(cfg))
    agent.control_structure = cfg.get("control_structure", "fixed_sp_mpc")
    return agent


def _oracle_factory(model=None, scenario=None, config=None, policy=None):
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


def _policy_factory(model=None, scenario=None, config=None, policy=None):
    cfg = dict(config or {})
    params = _controller_params(cfg)
    pol = policy or params.pop("policy", None)
    if pol is None:
        raise ValueError("policy controller requires a policy object")
    params.setdefault("action_mode", cfg.get("action_mode", "actuator"))
    params.setdefault("control_structure", cfg.get("control_structure", "learned_policy"))
    return PolicyController(pol, **params)


def _sb3_factory(model=None, scenario=None, config=None, policy=None):
    cfg = dict(config or {})
    params = _controller_params(cfg)
    params.setdefault("action_mode", cfg.get("action_mode", "setpoint"))
    params.setdefault("control_structure", cfg.get("control_structure", "sb3_policy"))
    # ``algo`` selects the loader for on-disk checkpoints. It is not a
    # PolicyController constructor argument when evaluating an in-memory model.
    algo = params.pop("algo", "sac")
    pol = policy or params.pop("policy", None)
    if pol is not None:
        params.setdefault("name", f"SB3-{algo.upper()}")
        return SB3PolicyController(pol, **params)
    path = params.pop("path")
    return SB3PolicyController.load(path, algo=algo, **params)


def _onnx_factory(model=None, scenario=None, config=None, policy=None):
    from .onnx import ONNXPolicyController

    cfg = dict(config or {})
    params = _controller_params(cfg)
    path = params.pop("path", None)
    if not path:
        raise ValueError("onnx controller requires a policy path")
    action_mode = params.pop("action_mode", cfg.get("action_mode", "setpoint"))
    normalized_actions = bool(params.pop("normalized_actions", False))
    params.setdefault("name", cfg.get("name", "ONNX-policy"))
    params.setdefault("control_structure", cfg.get("control_structure", "onnx_policy"))
    expected_action_dim = (
        len(getattr(model, "supervisory_layout", ()))
        if action_mode == "setpoint"
        else model.action_dim()
    )
    return ONNXPolicyController.load(
        path,
        action_mode=action_mode,
        expected_action_dim=expected_action_dim,
        scenario=scenario or model.scenario,
        normalized_actions=normalized_actions,
        **params,
    )


register_controller("hold", _hold_factory)
register_controller("pid", _pid_factory)
register_controller("mpc", _mpc_factory)
register_controller("oracle", _oracle_factory)
register_controller("policy", _policy_factory)
register_controller("sb3", _sb3_factory)
register_controller("onnx", _onnx_factory)
BUILTIN_CONTROLLERS.update(_REGISTRY)

"""Scenario-neutral rollout recording."""
from __future__ import annotations

from ..._internal.serialization import jsonable as _jsonable
from ...controllers import as_controller, build_context, validate_action
from ..results import result_schema
from .metadata import _env_disturbances, _env_metadata, _env_objective
from ..objective_specs import ObjectiveSpec, objective_spec
from ..protocols import BenchmarkProtocol


def rollout_controller(agent, env, seed: int = 0, max_steps: int | None = None,
                       protocol: BenchmarkProtocol | None = None,
                       objective_specification: ObjectiveSpec | None = None):
    """Run one episode and return a generic per-step rollout artifact.

    The recorder is scenario-neutral. Common fields are always present, and
    scenario-specific data from env ``info`` is preserved under each step.
    """

    controller = as_controller(agent, action_mode=getattr(env, "action_mode", "actuator"))
    obs, reset_info = env.reset(seed=seed)
    controller.reset(seed=seed)
    rows = []
    info = reset_info or {}
    done = False
    step = 0
    limit = max_steps if max_steps is not None else getattr(env, "episode_steps", None)
    while not done and (limit is None or step < limit):
        context = build_context(env, info)
        action = validate_action(controller.act(obs, context), env, controller.name)
        state = list(getattr(env.integ, "x", []))
        obs_next, reward, term, trunc, info_next = env.step(action)
        rows.append(_jsonable({
            "step": step,
            "time": step * float(env.control_dt),
            "obs": obs,
            "state": state,
            "action": action,
            "setpoint": context.setpoint,
            "measurement": context.measurement,
            "disturbance": _env_disturbances(env),
            "reward": reward,
            "profit": info_next.get("profit", 0.0),
            "constraint": info_next.get("constraint", 0.0),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "next_obs": obs_next,
            "next_state": list(getattr(env.integ, "x", [])),
            "info": info_next,
        }))
        obs = obs_next
        info = info_next
        done = bool(term or trunc)
        step += 1

    return {
        "controller_name": controller.name,
        "seed": int(seed),
        "steps": len(rows),
        "objective": (
            protocol.objective
            if protocol is not None
            else getattr(objective_specification, "name", _env_objective(env))
        ),
        "objective_source": (
            protocol.objective_source
            if protocol is not None
            else getattr(objective_specification, "source", "environment-reward-mode")
        ),
        "protocol": (
            protocol.metadata()
            if protocol is not None
            else {
                "environment": _env_metadata(env),
                "objective_spec": (
                    objective_specification.metadata()
                    if objective_specification is not None
                    else objective_spec(_env_objective(env), source="environment-reward-mode").metadata()
                ),
            }
        ),
        "controller": controller.metadata(),
        "setpoint_schedule": _jsonable(
            getattr(env, "_episode_setpoint_events", {})
        ),
        "scorer": _jsonable(env.scorer.report()),
        "rollout_schema": result_schema()["rollout"],
        "rollout": rows,
    }

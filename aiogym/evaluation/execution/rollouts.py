"""Scenario-neutral rollout recording."""
from __future__ import annotations

from ..._internal.serialization import jsonable as _jsonable
from ...controllers.adapters import as_controller
from ...controllers.contracts import build_context, validate_action
from ..results import result_schema
from .metadata import _env_disturbances, _env_metadata


def _rollout_step(
    *,
    step,
    env,
    obs,
    state,
    action,
    context,
    obs_next,
    reward,
    term,
    trunc,
    info_next,
):
    return _jsonable({
        "step": step,
        "time": step * float(env.control_dt),
        "obs": obs,
        "state": state,
        "action": action,
        "setpoint": context.setpoint,
        "measurement": context.measurement,
        "disturbance": _env_disturbances(env),
        "reward": reward,
        "reward_spec_id": str(
            info_next.get(
                "reward_spec_id",
                getattr(env, "reward_spec_id", "unknown"),
            )
        ),
        "profit": info_next.get("profit", 0.0),
        "constraint": info_next.get("constraint", 0.0),
        "terminated": bool(term),
        "truncated": bool(trunc),
        "next_obs": obs_next,
        "next_state": list(getattr(env.integ, "x", [])),
        "info": info_next,
    })


def _rollout_payload(
    controller,
    env,
    *,
    seed,
    rows,
):
    return {
        "controller_name": controller.name,
        "seed": int(seed),
        "steps": len(rows),
        "goal": str(getattr(env, "goal", "regulation")),
        "reward_spec_id": str(
            getattr(env, "reward_spec_id", "unknown")
        ),
        "return_comparable_across_reward_specs": False,
        "environment": _env_metadata(env),
        "controller": controller.metadata(),
        "setpoint_schedule": _jsonable(
            getattr(env, "_episode_setpoint_events", {})
        ),
        "rollout_schema": result_schema()["rollout"],
        "rollout": rows,
    }


def rollout_controller(agent, env, seed: int = 0, max_steps: int | None = None):
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
        rows.append(_rollout_step(
            step=step,
            env=env,
            obs=obs,
            state=state,
            action=action,
            context=context,
            obs_next=obs_next,
            reward=reward,
            term=term,
            trunc=trunc,
            info_next=info_next,
        ))
        obs = obs_next
        info = info_next
        done = bool(term or trunc)
        step += 1

    return _rollout_payload(
        controller,
        env,
        seed=seed,
        rows=rows,
    )

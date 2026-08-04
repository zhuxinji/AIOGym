"""Internal Track/Case builders consuming resolved specifications."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from aiogym._environment.factory import make_env


def build_case_environment(
    scenario,
    case,
    reward_spec,
    *,
    info_level: str = "full",
):
    return make_env(
        config={
            "scenario": scenario,
            "case": case.profile,
            "reward_spec": reward_spec,
            "info_level": info_level,
            "environment": {},
        }
    )


def build_track_case_environment(
    track,
    case,
    *,
    info_level: str = "full",
    profile_timing: bool = False,
):
    environment = {
        name: track.policy_contract[name]
        for name in (
            "action_mode",
            "control_dt",
            "disturbance_obs",
            "previous_action_obs",
            "normalize_observations",
            "tracking_error_obs",
            "integral_obs",
            "observation_mode",
        )
        if name in track.policy_contract
    }
    if profile_timing:
        environment["profile_timing"] = True
    env = make_env(
        config={
            "scenario": track.scenario,
            "case": case.profile,
            "reward_spec": track.reward_spec_id,
            "info_level": info_level,
            "environment": environment,
        }
    )
    if (
        track.scenario == "quadruple"
        and track.id.endswith("-regulation-generalist-v2")
    ):
        _freeze_physical_observation_bounds(env)
    from aiogym.rl.observations import (
        ObservationContract,
        wrap_observation_contract,
    )

    contract = track.policy_contract
    observation_contract = ObservationContract(
        sensing=contract.get("observation_mode", "full_state"),
        temporal=contract.get("temporal_observation", "single_step"),
        history_length=contract.get("history_length", 1),
        include_action_history=contract.get(
            "include_action_history",
            False,
        ),
        recurrent_state_shape=tuple(
            contract.get("recurrent_state_shape", ())
        ),
    )
    return wrap_observation_contract(env, observation_contract)


def _freeze_physical_observation_bounds(env) -> None:
    state_schema = (
        env.model.state_schema()
        if env.observation_mode == "full_state"
        else env.model.setpoint_schema()
    )
    lows, highs = _schema_bounds(state_schema, delta=False)
    reference_low, reference_high = _schema_bounds(
        env.model.setpoint_schema(),
        delta=env.tracking_error_obs,
    )
    lows.extend(reference_low)
    highs.extend(reference_high)
    if env.disturbance_obs:
        disturbance_schema = {
            row.get("name"): row for row in env.model.disturbance_schema()
        }
        extra_low, extra_high = _schema_bounds(
            [
                disturbance_schema.get(name, {})
                for name in env.model.dynamics_disturbance_names()
            ],
            delta=False,
        )
        lows.extend(extra_low)
        highs.extend(extra_high)
    if env.previous_action_obs:
        extra_low, extra_high = _schema_bounds(
            env.model.action_schema(),
            delta=False,
        )
        lows.extend(extra_low)
        highs.extend(extra_high)
    if env.integral_obs:
        lows.extend([-1.0] * len(env.model.setpoint_schema()))
        highs.extend([1.0] * len(env.model.setpoint_schema()))
    if len(lows) != int(np.prod(env.observation_space.shape)):
        raise ValueError("v2 physical observation bounds do not match shape")
    env.observation_space = spaces.Box(
        np.asarray(lows, dtype=np.float32),
        np.asarray(highs, dtype=np.float32),
        dtype=np.float32,
    )


def _schema_bounds(schema, *, delta: bool):
    lows = []
    highs = []
    for row in schema:
        bounds = row.get("bounds") if isinstance(row, dict) else None
        if (
            not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or bounds[0] is None
            or bounds[1] is None
        ):
            raise ValueError("v2 observation schema requires finite bounds")
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("v2 observation schema bounds are invalid")
        if delta:
            span = high - low
            lows.append(-span)
            highs.append(span)
        else:
            lows.append(low)
            highs.append(high)
    return lows, highs


__all__ = [
    "build_case_environment",
    "build_track_case_environment",
]

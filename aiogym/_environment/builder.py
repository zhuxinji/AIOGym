"""Internal Track/Case builders consuming resolved specifications."""
from __future__ import annotations

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


__all__ = [
    "build_case_environment",
    "build_track_case_environment",
]

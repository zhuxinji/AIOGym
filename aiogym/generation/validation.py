"""Validation at the EpisodeSpec/environment execution boundary."""
from __future__ import annotations

import math

from aiogym._environment.realism import (
    validate_actuator_model,
    validate_sensor_model,
)
from aiogym._internal.identifiers import canonical_scenario_id

from .specs import EpisodeSpec


def validate_episode_for_env(episode: EpisodeSpec, env) -> None:
    """Reject a resolved episode that cannot execute in ``env`` unchanged."""

    if not isinstance(episode, EpisodeSpec):
        raise TypeError("episode must be an EpisodeSpec")
    if canonical_scenario_id(episode.scenario_id) != canonical_scenario_id(
        env.scenario
    ):
        raise ValueError(
            f"EpisodeSpec scenario {episode.scenario_id!r} does not match "
            f"environment scenario {env.scenario!r}"
        )
    if episode.goal != env.goal:
        raise ValueError(
            f"EpisodeSpec goal {episode.goal!r} does not match environment "
            f"goal {env.goal!r}"
        )
    if not math.isclose(
        episode.control_dt,
        float(env.control_dt),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"EpisodeSpec control_dt {episode.control_dt} does not match "
            f"environment control_dt {env.control_dt}"
        )
    if episode.episode_steps != int(env.episode_steps):
        raise ValueError(
            f"EpisodeSpec episode_steps {episode.episode_steps} does not "
            f"match environment episode_steps {env.episode_steps}"
        )
    expected_state_dim = len(env.model.initial_state())
    if len(episode.initial_state) != expected_state_dim:
        raise ValueError(
            f"EpisodeSpec initial_state has length "
            f"{len(episode.initial_state)}; expected {expected_state_dim}"
        )
    parameters = episode.plant_parameters
    missing_parameters = sorted(set(env._p_nominal) - set(parameters))
    unknown_parameters = sorted(set(parameters) - set(env._p_nominal))
    if missing_parameters or unknown_parameters:
        details = []
        if missing_parameters:
            details.append("missing: " + ", ".join(missing_parameters))
        if unknown_parameters:
            details.append("unknown: " + ", ".join(unknown_parameters))
        raise ValueError(
            "EpisodeSpec plant_parameters must be a complete model snapshot ("
            + "; ".join(details)
            + ")"
        )

    reference_schedule = episode.reference_schedule
    if not reference_schedule or int(
        reference_schedule[0].get("at_step", -1)
    ) != 0:
        raise ValueError(
            "EpisodeSpec reference_schedule must resolve the initial "
            "reference at step 0"
        )
    reference_dim = len(env._ysp0)
    for event in reference_schedule:
        values = event.get("values")
        if not isinstance(values, list) or len(values) != reference_dim:
            raise ValueError(
                "EpisodeSpec reference event values must have length "
                f"{reference_dim}"
            )

    for event in episode.disturbance_schedule:
        if not {"at_step", "name", "value"} <= set(event):
            raise ValueError(
                "EpisodeSpec disturbance events require at_step, name, value"
            )
        env._validate_case_disturbance(event["name"], event["value"])

    validate_sensor_model(
        episode.sensor_model,
        dimension=expected_state_dim,
    )
    validate_actuator_model(
        episode.actuator_model,
        dimension=int(env.nu),
    )


__all__ = ["validate_episode_for_env"]

"""Episode factories and formal Benchmarks for the fired heater."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._metrics import regulation_episode_metrics


_TRACKING_OXYGEN_RANGE = (2.5, 5.0)
_TRACKING_TEMPERATURE_RANGE_C = (364.0, 372.0)
_MINIMUM_OXYGEN_MOVE = 0.4
_MINIMUM_TEMPERATURE_MOVE_C = 2.0
_TRACKING_ACTION_RANGE = (0.05, 0.95)
_BOUNDARY_INITIAL_PROBABILITY = 0.20
_MAXIMUM_SAMPLING_ATTEMPTS = 100
_TRACKING_HORIZON = 300


def make_default_episode(model) -> EpisodeSpec:
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=(4.0, 366.0),
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _tracking_episode(model, rng) -> EpisodeSpec:
    start = _sample_tracking_equilibrium(model, rng)
    target = _sample_tracking_equilibrium(
        model,
        rng,
        previous_reference=start["reference"],
    )
    return EpisodeSpec(
        initial_state=start["state"],
        initial_action=start["action"],
        reference=target["reference"],
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _sample_tracking_equilibrium(model, rng, *, previous_reference=None):
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        reference = np.asarray(
            [
                rng.uniform(*_TRACKING_OXYGEN_RANGE),
                rng.uniform(*_TRACKING_TEMPERATURE_RANGE_C),
            ],
            dtype=float,
        )
        if previous_reference is not None:
            difference = np.abs(reference - previous_reference)
            if (
                difference[0] < _MINIMUM_OXYGEN_MOVE
                or difference[1] < _MINIMUM_TEMPERATURE_MOVE_C
            ):
                continue
        action = model.tracking_steady_state_action(reference)
        state = model.tracking_steady_state_state(reference)
        if action is None or state is None:
            continue
        action_values = np.asarray(action, dtype=float)
        lower, upper = _TRACKING_ACTION_RANGE
        derivative = np.asarray(
            model.dynamics(state, action, model.default_disturbances()),
            dtype=float,
        )
        if (
            np.any(action_values < lower)
            or np.any(action_values > upper)
            or not np.allclose(model.outputs(state), reference, rtol=0.0, atol=1e-10)
            or not np.allclose(derivative, 0.0, rtol=0.0, atol=1e-10)
        ):
            continue
        return {
            "state": tuple(float(value) for value in state),
            "action": tuple(float(value) for value in action),
            "reference": tuple(float(value) for value in reference),
        }
    raise ValueError(
        "could not sample a feasible heater tracking equilibrium within "
        f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    defaults = model.default_disturbances()
    start = int(rng.integers(140, 241))
    duration = int(rng.integers(200, 321))
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        disturbed = {
            "feed_temperature": float(rng.uniform(255.0, 305.0)),
            "ambient_temperature": float(rng.uniform(5.0, 35.0)),
            "feed_flow": float(rng.uniform(75.0, 105.0)),
            "fuel_heating_value_factor": float(rng.uniform(0.85, 1.15)),
        }
        action = model.tracking_steady_state_action(
            model.default_setpoint_vector(),
            disturbed,
        )
        state = model.tracking_steady_state_state(
            model.default_setpoint_vector(),
            disturbed,
        )
        if action is None or state is None:
            continue
        action_values = np.asarray(action, dtype=float)
        lower, upper = _TRACKING_ACTION_RANGE
        if np.any(action_values < lower) or np.any(action_values > upper):
            continue
        break
    else:
        raise ValueError(
            "could not sample a feasible heater disturbance within "
            f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
        )
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=700,
        disturbances=defaults,
        disturbance_schedule={
            start: disturbed,
            start + duration: defaults,
        },
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    boundary_reference = (
        float(rng.uniform(1.30, 1.70)),
        float(rng.uniform(382.0, 390.0)),
    )
    initial_state = model.tracking_steady_state_state(boundary_reference)
    initial_action = model.tracking_steady_state_action(boundary_reference)
    if initial_state is None or initial_action is None:
        raise ValueError("heater boundary operating point is infeasible")
    return EpisodeSpec(
        initial_state=tuple(initial_state),
        initial_action=tuple(initial_action),
        reference=tuple(model.default_setpoint_vector()),
        horizon=450,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(model, rng, reward_id) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(f"unsupported heater training reward {reward_id!r}")
    target = _sample_tracking_equilibrium(model, rng)
    if rng.random() < _BOUNDARY_INITIAL_PROBABILITY:
        boundary_reference = (
            float(rng.uniform(1.5, 1.7)),
            float(rng.uniform(382.0, 388.0)),
        )
        initial_state = model.tracking_steady_state_state(boundary_reference)
        initial_action = model.tracking_steady_state_action(boundary_reference)
        if initial_state is None or initial_action is None:
            raise ValueError(
                "sampled heater boundary operating point is infeasible"
            )
        episode = EpisodeSpec(
            initial_state=tuple(initial_state),
            initial_action=tuple(initial_action),
            reference=target["reference"],
            horizon=_TRACKING_HORIZON,
            disturbances=model.default_disturbances(),
        )
    else:
        start = _sample_tracking_equilibrium(
            model,
            rng,
            previous_reference=target["reference"],
        )
        episode = EpisodeSpec(
            initial_state=start["state"],
            initial_action=start["action"],
            reference=target["reference"],
            horizon=_TRACKING_HORIZON,
            disturbances=model.default_disturbances(),
        )
    return episode, "tracking"


def sample_training_disturbance(model, rng):
    defaults = model.default_disturbances()
    start = int(rng.integers(70, 121))
    duration = int(rng.integers(100, 161))
    return {
        start: {
            "feed_temperature": float(rng.uniform(255.0, 305.0)),
            "ambient_temperature": float(rng.uniform(5.0, 35.0)),
            "feed_flow": float(rng.uniform(75.0, 105.0)),
            "fuel_heating_value_factor": float(rng.uniform(0.85, 1.15)),
        },
        start + duration: defaults,
    }


BENCHMARKS = {
    "tracking": Benchmark(
        id="tracking",
        reward_id="regulation",
        episode_factory=_tracking_episode,
        metric_function=regulation_episode_metrics,
        ranking_metrics=(
            ("unsafe_rate", "minimize"),
            ("return", "maximize"),
        ),
    ),
    "disturbance-rejection": Benchmark(
        id="disturbance-rejection",
        reward_id="regulation",
        episode_factory=_disturbance_episode,
        metric_function=regulation_episode_metrics,
        ranking_metrics=(
            ("unsafe_rate", "minimize"),
            ("return", "maximize"),
        ),
    ),
    "boundary-safety": Benchmark(
        id="boundary-safety",
        reward_id="regulation",
        episode_factory=_boundary_episode,
        metric_function=regulation_episode_metrics,
        ranking_metrics=(
            ("unsafe_rate", "minimize"),
            ("return", "maximize"),
        ),
    ),
}


__all__ = [
    "BENCHMARKS",
    "make_default_episode",
    "sample_training_disturbance",
    "sample_training_episode",
]

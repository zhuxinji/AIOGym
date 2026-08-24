"""Episode factories and formal Benchmarks for multistage extraction."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._metrics import regulation_episode_metrics


_TRACKING_RANGE = (0.12, 0.44)
_MINIMUM_TRACKING_MOVE = 0.05
_TRACKING_ACTION_RANGE = (0.05, 0.95)
_BOUNDARY_INITIAL_PROBABILITY = 0.20
_MAXIMUM_SAMPLING_ATTEMPTS = 100
_TRACKING_HORIZON = 100


def make_default_episode(model) -> EpisodeSpec:
    target_reference = (0.40,)
    if model.tracking_steady_state_action(target_reference) is None:
        raise ValueError("extraction default tracking target is infeasible")
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=target_reference,
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _tracking_episode(model, rng) -> EpisodeSpec:
    target = _sample_tracking_equilibrium(
        model,
        rng,
        previous_reference=tuple(model.default_setpoint_vector()),
    )
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=target["reference"],
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _sample_tracking_equilibrium(model, rng, *, previous_reference=None):
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        reference = np.asarray([rng.uniform(*_TRACKING_RANGE)], dtype=float)
        if previous_reference is not None and np.any(
            np.abs(reference - previous_reference) < _MINIMUM_TRACKING_MOVE
        ):
            continue
        action = model.tracking_steady_state_action(reference)
        state = model.tracking_steady_state_state(reference)
        if action is None or state is None:
            continue
        action_values = np.asarray(action, dtype=float)
        state_values = np.asarray(state, dtype=float)
        lower, upper = _TRACKING_ACTION_RANGE
        derivative = np.asarray(
            model.dynamics(state, action, model.default_disturbances()),
            dtype=float,
        )
        if (
            np.any(action_values < lower)
            or np.any(action_values > upper)
            or np.any(state_values < 0.0)
            or np.any(state_values > 1.0)
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
        "could not sample a feasible extraction tracking equilibrium within "
        f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    defaults = model.default_disturbances()
    start = int(rng.integers(140, 241))
    duration = int(rng.integers(200, 321))
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=580,
        disturbances=defaults,
        disturbance_schedule={
            start: {
                "liquid_feed_concentration": float(rng.uniform(0.45, 0.75)),
                "gas_feed_concentration": float(rng.uniform(0.02, 0.09)),
                "mass_transfer_coefficient": float(rng.uniform(3.5, 6.5)),
            },
            start + duration: defaults,
        },
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    boundary_reference = (float(rng.uniform(0.01, 0.04)),)
    initial_state = model.tracking_steady_state_state(boundary_reference)
    initial_action = model.tracking_steady_state_action(boundary_reference)
    if initial_state is None or initial_action is None:
        raise ValueError("extraction boundary operating point is infeasible")
    return EpisodeSpec(
        initial_state=tuple(initial_state),
        initial_action=tuple(initial_action),
        reference=tuple(model.default_setpoint_vector()),
        horizon=100,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(model, rng, reward_id) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(f"unsupported extraction training reward {reward_id!r}")
    target = _sample_tracking_equilibrium(
        model,
        rng,
        previous_reference=tuple(model.default_setpoint_vector()),
    )
    if rng.random() < _BOUNDARY_INITIAL_PROBABILITY:
        boundary_reference = (float(rng.uniform(0.01, 0.04)),)
        initial_state = model.tracking_steady_state_state(boundary_reference)
        initial_action = model.tracking_steady_state_action(boundary_reference)
        if initial_state is None or initial_action is None:
            raise ValueError(
                "sampled extraction boundary operating point is infeasible"
            )
        episode = EpisodeSpec(
            initial_state=tuple(initial_state),
            initial_action=tuple(initial_action),
            reference=target["reference"],
            horizon=_TRACKING_HORIZON,
            disturbances=model.default_disturbances(),
        )
    else:
        episode = EpisodeSpec(
            initial_state=tuple(model.initial_state()),
            initial_action=tuple(model.default_action()),
            reference=target["reference"],
            horizon=_TRACKING_HORIZON,
            disturbances=model.default_disturbances(),
        )
    return episode, "tracking"


def sample_training_disturbance(model, rng):
    defaults = model.default_disturbances()
    start = int(rng.integers(23, 41))
    duration = int(rng.integers(33, 54))
    return {
        start: {
            "liquid_feed_concentration": float(rng.uniform(0.45, 0.75)),
            "gas_feed_concentration": float(rng.uniform(0.02, 0.09)),
            "mass_transfer_coefficient": float(rng.uniform(3.5, 6.5)),
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

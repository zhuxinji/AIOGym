"""Episode factories and formal benchmarks for the two-input CSTR."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._boundary import forward_preroll
from aiogym.scenarios._episodes import tracking_episode
from aiogym.scenarios._metrics import regulation_episode_metrics


_TRACKING_CONCENTRATION_RANGE = (0.02, 0.20)
_TRACKING_TEMPERATURE_RANGE_C = (50.0, 82.0)
_MINIMUM_TRACKING_MOVE = np.asarray([0.02, 5.0], dtype=float)
_BOUNDARY_TEMPERATURE_RANGE_C = (86.0, 90.0)
_TRACKING_ACTION_RANGE = (0.05, 0.95)
_MAXIMUM_SAMPLING_ATTEMPTS = 100
_TRACKING_HORIZON = 225

_DISTURBANCE_HORIZON = 400
_BOUNDARY_HORIZON = 100


def make_default_episode(model) -> EpisodeSpec:
    target_reference = (0.075, 72.0)
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=target_reference,
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _tracking_episode(model, rng) -> EpisodeSpec:
    return tracking_episode(
        model, rng, sample_equilibrium=_sample_tracking_equilibrium,
        horizon=_TRACKING_HORIZON,
    )


def _sample_tracking_equilibrium(model, rng, *, previous_reference=None):
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        reference = np.asarray(
            [
                rng.uniform(*_TRACKING_CONCENTRATION_RANGE),
                rng.uniform(*_TRACKING_TEMPERATURE_RANGE_C),
            ],
            dtype=float,
        )
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
            or np.any(state_values < np.asarray([0.0, 0.0]))
            or np.any(state_values > np.asarray([1.5, 200.0]))
            or not np.allclose(model.outputs(state), reference, rtol=0.0, atol=1e-12)
            or not np.allclose(derivative, 0.0, rtol=0.0, atol=1e-12)
        ):
            continue
        return {
            "state": tuple(float(value) for value in state),
            "action": tuple(float(value) for value in action),
            "reference": tuple(float(value) for value in reference),
        }
    raise ValueError(
        "could not sample a feasible CSTR tracking equilibrium within "
        f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    defaults = model.default_disturbances()
    start = int(rng.integers(100, 181))
    # Restore by 80%: leave 10% recovery before the final 10% stability window.
    duration = int(rng.integers(100, min(180, 320 - start) + 1))
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=_DISTURBANCE_HORIZON,
        disturbances=defaults,
        disturbance_schedule={
            start: {
                "feed_temperature": float(rng.uniform(12.0, 32.0)),
                "feed_concentration": float(rng.uniform(0.75, 1.30)),
                "coolant_temperature": float(rng.uniform(5.0, 20.0)),
            },
            start + duration: defaults,
        },
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    boundary = _sample_boundary_preroll(model, rng)
    return EpisodeSpec(
        initial_state=boundary["state"],
        initial_action=boundary["action"],
        reference=tuple(model.default_setpoint_vector()),
        horizon=_BOUNDARY_HORIZON,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(
    model, rng, reward_id, boundary: bool
) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(f"unsupported CSTR training reward {reward_id!r}")
    target = _sample_tracking_equilibrium(model, rng)
    if boundary:
        boundary_case = _sample_boundary_preroll(model, rng)
        episode = EpisodeSpec(
            initial_state=boundary_case["state"],
            initial_action=boundary_case["action"],
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
    return episode, "boundary-prerun" if boundary else "interior"


def _sample_boundary_preroll(model, rng):
    target_temperature = float(rng.uniform(*_BOUNDARY_TEMPERATURE_RANGE_C))
    return forward_preroll(
        model,
        command=(float(rng.uniform(0.95, 1.0)), float(rng.uniform(0.0, 0.05))),
        reached=lambda state: float(state[1]) >= target_temperature,
        control_dt=1.0,
        maximum_steps=50,
    )


def sample_training_disturbance(model, rng):
    defaults = model.default_disturbances()
    start = int(rng.integers(50, 91))
    duration = int(rng.integers(50, 91))
    return {
        start: {
            "feed_temperature": float(rng.uniform(12.0, 32.0)),
            "feed_concentration": float(rng.uniform(0.75, 1.30)),
            "coolant_temperature": float(rng.uniform(5.0, 20.0)),
        },
        start + duration: defaults,
    }


BENCHMARKS = {
    "tracking": Benchmark(
        id="tracking",
        description='Track feasible output targets from sampled operating points.',
        horizon=_TRACKING_HORIZON,
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
        description='Reject scheduled physical disturbances while maintaining the output targets.',
        horizon=_DISTURBANCE_HORIZON,
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
        description='Recover safely from initial conditions near scenario safety boundaries.',
        horizon=_BOUNDARY_HORIZON,
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

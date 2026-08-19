"""Episode factories and formal benchmarks for the Quadruple-Tank scenario."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._metrics import regulation_episode_metrics


_TRACKING_LEVEL_RANGE_FRACTION = (0.35, 0.80)
_TRACKING_ACTION_RANGE = (0.02, 0.95)
_MINIMUM_TRACKING_MOVE_FRACTION = 0.15
_BOUNDARY_INITIAL_PROBABILITY = 0.20
_BOUNDARY_LEVEL_RANGE_FRACTION = (0.80, 0.92)
_MAXIMUM_SAMPLING_ATTEMPTS = 100


def make_default_episode(model) -> EpisodeSpec:
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=(18.0, 8.0),
        horizon=600,
        disturbances=model.default_disturbances(),
        reference_schedule={300: (8.0, 18.0)},
    )


def _tracking_episode(model, rng) -> EpisodeSpec:
    start = _sample_tracking_equilibrium(model, rng)
    first = _sample_tracking_equilibrium(
        model,
        rng,
        previous_reference=start["reference"],
    )
    second = _sample_tracking_equilibrium(
        model,
        rng,
        previous_reference=first["reference"],
    )
    return EpisodeSpec(
        initial_state=start["state"],
        initial_action=start["action"],
        reference=start["reference"],
        horizon=600,
        disturbances=model.default_disturbances(),
        reference_schedule={
            120: first["reference"],
            360: second["reference"],
        },
    )


def _sample_tracking_equilibrium(model, rng, *, previous_reference=None):
    maximum = float(model.parameter("max_level"))
    lower, upper = (
        fraction * maximum for fraction in _TRACKING_LEVEL_RANGE_FRACTION
    )
    minimum_move = _MINIMUM_TRACKING_MOVE_FRACTION * maximum
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        reference = rng.uniform(lower, upper, size=2)
        if previous_reference is not None and np.any(
            np.abs(reference - previous_reference) < minimum_move
        ):
            continue
        action = model.tracking_steady_state_action(reference)
        state = model.tracking_steady_state_state(reference)
        if action is None or state is None:
            continue
        action_array = np.asarray(action, dtype=float)
        state_array = np.asarray(state, dtype=float)
        action_lower, action_upper = _TRACKING_ACTION_RANGE
        if (
            np.any(action_array < action_lower)
            or np.any(action_array > action_upper)
            or np.any(state_array < 0.0)
            or np.any(state_array > maximum)
            or not np.allclose(
                model.outputs(state_array),
                reference,
                rtol=1e-9,
                atol=1e-9,
            )
            or not np.allclose(
                model.dynamics(
                    state_array,
                    action_array,
                    model.default_disturbances(),
                ),
                0.0,
                rtol=0.0,
                atol=1e-10,
            )
        ):
            continue
        return {
            "state": tuple(float(value) for value in state_array),
            "action": tuple(float(value) for value in action_array),
            "reference": tuple(float(value) for value in reference),
        }
    raise ValueError(
        "could not sample a feasible Quadruple-Tank tracking equilibrium "
        f"within {_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    del rng
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=600,
        disturbances=model.default_disturbances(),
        disturbance_schedule={
            150: {"pump_flow_factor": 0.85},
            350: {"pump_flow_factor": 1.0},
        },
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    del rng
    maximum = float(model.parameter("max_level"))
    initial = (0.9 * maximum,) * 4
    reference = np.clip(
        np.asarray(model.default_setpoint_vector(), dtype=float), 0.0, maximum
    )
    return EpisodeSpec(
        initial_state=initial,
        initial_action=tuple(model.default_action()),
        reference=tuple(reference),
        horizon=400,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(model, rng, reward_id) -> tuple[EpisodeSpec, str]:
    del reward_id
    maximum = float(model.parameter("max_level"))
    boundary_initial = rng.random() < _BOUNDARY_INITIAL_PROBABILITY
    target = _sample_tracking_equilibrium(model, rng)
    if boundary_initial:
        initial_state = tuple(
            float(value)
            for value in rng.uniform(
                _BOUNDARY_LEVEL_RANGE_FRACTION[0] * maximum,
                _BOUNDARY_LEVEL_RANGE_FRACTION[1] * maximum,
                size=4,
            )
        )
        episode = EpisodeSpec(
            initial_state=initial_state,
            initial_action=target["action"],
            reference=target["reference"],
            horizon=600,
            disturbances=model.default_disturbances(),
        )
    else:
        start = _sample_tracking_equilibrium(
            model,
            rng,
            previous_reference=target["reference"],
        )
        step = int(rng.integers(90, 241))
        episode = EpisodeSpec(
            initial_state=start["state"],
            initial_action=start["action"],
            reference=start["reference"],
            horizon=600,
            disturbances=model.default_disturbances(),
            reference_schedule={step: target["reference"]},
        )
    return episode, "tracking"


def sample_training_disturbance(model, rng):
    del model
    start = int(rng.integers(120, 241))
    duration = int(rng.integers(120, 241))
    factor = float(rng.uniform(0.78, 0.94))
    return {
        start: {"pump_flow_factor": factor},
        start + duration: {"pump_flow_factor": 1.0},
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

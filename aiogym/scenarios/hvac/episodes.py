"""Episode factories and formal benchmarks for the two-zone HVAC scenario."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._boundary import forward_preroll
from aiogym.scenarios._episodes import tracking_episode
from aiogym.scenarios._metrics import regulation_episode_metrics


_TRACKING_TEMPERATURE_RANGE_C = (19.0, 25.0)
_MINIMUM_TRACKING_MOVE_C = 1.5
_TRACKING_ACTION_RANGE = (0.05, 0.95)
_MAXIMUM_SAMPLING_ATTEMPTS = 100
_TRACKING_HORIZON = 60

_DISTURBANCE_HORIZON = 210
_BOUNDARY_HORIZON = 60


def make_default_episode(model) -> EpisodeSpec:
    target_reference = (24.0, 20.0)
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
        reference = rng.uniform(*_TRACKING_TEMPERATURE_RANGE_C, size=2)
        if previous_reference is not None and np.any(
            np.abs(reference - previous_reference) < _MINIMUM_TRACKING_MOVE_C
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
        "could not sample a feasible HVAC tracking equilibrium within "
        f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    defaults = model.default_disturbances()
    start = int(rng.integers(50, 91))
    duration = int(rng.integers(40, 81))
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=_DISTURBANCE_HORIZON,
        disturbances=defaults,
        disturbance_schedule={
            start: {
                "outdoor_temperature": float(rng.uniform(-10.0, 5.0)),
                "internal_heat_load_zone_0": float(rng.uniform(300.0, 800.0)),
                "internal_heat_load_zone_1": float(rng.uniform(-500.0, -100.0)),
                "hvac_efficiency": float(rng.uniform(0.65, 0.90)),
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


def _sample_boundary_preroll(model, rng):
    disturbances = model.default_disturbances()
    if rng.random() < 0.5:
        target = float(rng.uniform(48.0, 54.0))
        command = tuple(float(value) for value in rng.uniform(0.95, 1.0, size=2))
        disturbances.update(
            {
                "outdoor_temperature": float(rng.uniform(48.0, 50.0)),
                "internal_heat_load_zone_0": float(rng.uniform(1800.0, 2000.0)),
                "internal_heat_load_zone_1": float(rng.uniform(1800.0, 2000.0)),
            }
        )
        reached = lambda state: bool(np.all(state >= target))
    else:
        target = float(rng.uniform(-14.0, -8.0))
        command = tuple(float(value) for value in rng.uniform(0.0, 0.05, size=2))
        disturbances.update(
            {
                "outdoor_temperature": float(rng.uniform(-30.0, -28.0)),
                "internal_heat_load_zone_0": float(rng.uniform(-1000.0, -800.0)),
                "internal_heat_load_zone_1": float(rng.uniform(-1000.0, -800.0)),
            }
        )
        reached = lambda state: bool(np.all(state <= target))
    return forward_preroll(
        model,
        command=command,
        reached=reached,
        control_dt=5.0,
        maximum_steps=30,
        disturbances=disturbances,
    )


def sample_training_episode(
    model, rng, reward_id, boundary: bool
) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(f"unsupported HVAC training reward {reward_id!r}")
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


def sample_training_disturbance(model, rng):
    defaults = model.default_disturbances()
    start = int(rng.integers(12, 23))
    duration = int(rng.integers(10, 21))
    return {
        start: {
            "outdoor_temperature": float(rng.uniform(-10.0, 15.0)),
            "internal_heat_load_zone_0": float(rng.uniform(-500.0, 800.0)),
            "internal_heat_load_zone_1": float(rng.uniform(-500.0, 800.0)),
            "hvac_efficiency": float(rng.uniform(0.7, 1.0)),
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

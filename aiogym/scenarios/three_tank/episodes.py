"""Episode factories and formal benchmarks for the Three-Tank scenario."""

from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from .metrics import regulation_episode_metrics


_TRACKING_LEVEL_RANGE_M = (0.125, 0.4)
_TRACKING_FLOW_RANGE_M3S = (1.0 / 60000.0, 6.0 / 60000.0)
_TRACKING_ACTION_RANGE = (0.02, 0.85)
_MINIMUM_LEVEL_MOVE_M = 0.05
_BOUNDARY_INITIAL_PROBABILITY = 0.20
_BOUNDARY_LEVEL_RANGE_FRACTION = (0.80, 0.92)
_MAXIMUM_SAMPLING_ATTEMPTS = 100
_TRACKING_HORIZON = 600


def make_default_episode(model) -> EpisodeSpec:
    return BENCHMARKS["tracking"].make_episode(model, 0)


def _sample_tracking_equilibria(model, rng):
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        flow = float(rng.uniform(*_TRACKING_FLOW_RANGE_M3S))
        start_levels = rng.uniform(*_TRACKING_LEVEL_RANGE_M, size=3)
        target_levels = np.empty(3, dtype=float)
        for index, start_level in enumerate(start_levels):
            directions = []
            downward_limit = start_level - _TRACKING_LEVEL_RANGE_M[0]
            upward_limit = _TRACKING_LEVEL_RANGE_M[1] - start_level
            if downward_limit >= _MINIMUM_LEVEL_MOVE_M:
                directions.append((-1.0, downward_limit))
            if upward_limit >= _MINIMUM_LEVEL_MOVE_M:
                directions.append((1.0, upward_limit))
            direction, limit = directions[int(rng.integers(len(directions)))]
            target_levels[index] = start_level + direction * float(
                rng.uniform(_MINIMUM_LEVEL_MOVE_M, limit)
            )
        start = _sampled_tracking_equilibrium(
            model,
            levels=start_levels,
            flow=flow,
        )
        target = _sampled_tracking_equilibrium(
            model,
            levels=target_levels,
            flow=flow,
        )
        if start is not None and target is not None:
            return start, target
    raise ValueError(
        "could not sample feasible Three-Tank tracking equilibria within "
        f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _sampled_tracking_equilibrium(model, *, levels, flow):
    result = model.nominal_steady_state(
        levels=levels,
        flow=flow,
    )
    action = np.asarray(result["action"], dtype=float)
    lower, upper = _TRACKING_ACTION_RANGE
    if not result["feasible"] or np.any(action < lower) or np.any(action > upper):
        return None
    return result


def _tracking_episode(model, rng) -> EpisodeSpec:
    start, target = _sample_tracking_equilibria(model, rng)
    return EpisodeSpec(
        initial_state=tuple(start["state"]),
        initial_action=tuple(start["action"]),
        reference=tuple(target["y_sp"]),
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    steady = model.nominal_steady_state()
    if not steady["feasible"]:
        raise ValueError(
            "Three-Tank disturbance benchmark equilibrium is infeasible for "
            f"the resolved parameters: {steady['infeasible_reasons']}"
        )
    hydraulic_start = int(rng.integers(600, 851))
    hydraulic_duration = int(rng.integers(400, 651))
    pump_factor = float(rng.uniform(0.40, 0.70))
    v23_factor = float(rng.uniform(0.50, 0.70))
    return EpisodeSpec(
        initial_state=tuple(steady["state"]),
        initial_action=tuple(steady["action"]),
        reference=tuple(steady["y_sp"]),
        horizon=1800,
        disturbances=model.default_disturbances(),
        disturbance_schedule={
            hydraulic_start: {
                "pump_flow_factor": pump_factor,
                "v23_flow_factor": v23_factor,
            },
            hydraulic_start + hydraulic_duration: {
                "pump_flow_factor": 1.0,
                "v23_flow_factor": 1.0,
            },
        },
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    maximum = np.asarray(model.parameter("height_max"), dtype=float)
    levels = rng.uniform(0.86, 0.90, size=3) * maximum
    return EpisodeSpec(
        initial_state=tuple(levels),
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=600,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(model, rng, reward_id) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(f"unsupported Three-Tank training reward {reward_id!r}")
    defaults = model.default_disturbances()
    boundary_initial = rng.random() < _BOUNDARY_INITIAL_PROBABILITY
    if boundary_initial:
        maximum = np.asarray(model.parameter("height_max"), dtype=float)
        boundary_lower = _BOUNDARY_LEVEL_RANGE_FRACTION[0] * maximum
        boundary_upper = _BOUNDARY_LEVEL_RANGE_FRACTION[1] * maximum
        start, target = _sample_tracking_equilibria(model, rng)
        levels = rng.uniform(boundary_lower, boundary_upper)
        episode = EpisodeSpec(
            initial_state=tuple(levels),
            initial_action=tuple(start["action"]),
            reference=tuple(target["y_sp"]),
            horizon=_TRACKING_HORIZON,
            disturbances=defaults,
        )
    else:
        start, target = _sample_tracking_equilibria(model, rng)
        episode = EpisodeSpec(
            initial_state=tuple(start["state"]),
            initial_action=tuple(start["action"]),
            reference=tuple(target["y_sp"]),
            horizon=_TRACKING_HORIZON,
            disturbances=defaults,
        )
    return episode, "tracking"


def sample_training_disturbance(model, rng):
    del model
    start = int(rng.integers(112, 201))
    duration = int(rng.integers(75, 151))
    factor = float(rng.uniform(0.72, 0.94))
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

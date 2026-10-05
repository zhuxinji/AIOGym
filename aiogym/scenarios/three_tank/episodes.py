"""Episode factories and formal benchmarks for the Three-Tank scenario."""

from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._boundary import forward_preroll
from .metrics import (
    disturbance_rejection_episode_metrics,
    regulation_episode_metrics,
)


_TRACKING_LEVEL_RANGE_M = (0.125, 0.4)
_TRACKING_FLOW_RANGE_M3S = (3.0 / 60000.0, 8.0 / 60000.0)
_TRACKING_ACTION_RANGE = (0.02, 0.85)
_MINIMUM_LEVEL_MOVE_M = 0.05
_BOUNDARY_LEVEL_RANGE_FRACTION = (0.82, 0.848)
_MAXIMUM_SAMPLING_ATTEMPTS = 100
_TRACKING_HORIZON = 600
_BYPASS_OPEN_STEP_RANGE = (30, 90)
_BYPASS_DURATION_STEPS = 360

_BOUNDARY_HORIZON = 600


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


def _sample_bypass_schedule(rng):
    bypass_mode = int(rng.integers(10))
    bypass_names = ("bv12_open", "bv23_open", "bv34_open")
    if bypass_mode < 3:
        active_names = (bypass_names[int(rng.integers(len(bypass_names)))],)
    elif bypass_mode < 7:
        omitted_index = int(rng.integers(len(bypass_names)))
        active_names = tuple(
            name for index, name in enumerate(bypass_names) if index != omitted_index
        )
    else:
        active_names = bypass_names
    bypass_open_step = int(
        rng.integers(
            _BYPASS_OPEN_STEP_RANGE[0],
            _BYPASS_OPEN_STEP_RANGE[1] + 1,
        )
    )
    bypass_close_step = bypass_open_step + _BYPASS_DURATION_STEPS
    return {
        bypass_open_step: {name: 1.0 for name in active_names},
        bypass_close_step: {name: 0.0 for name in active_names},
    }


def _disturbance_episode(model, rng) -> EpisodeSpec:
    start, target = _sample_tracking_equilibria(model, rng)
    return EpisodeSpec(
        initial_state=tuple(start["state"]),
        initial_action=tuple(start["action"]),
        reference=tuple(target["y_sp"]),
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
        disturbance_schedule=_sample_bypass_schedule(rng),
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
    maximum = np.asarray(model.parameter("height_max"), dtype=float)
    tank = int(rng.integers(3))
    target = float(rng.uniform(*_BOUNDARY_LEVEL_RANGE_FRACTION)) * maximum[tank]
    low = rng.uniform(0.03, 0.07, size=3)
    transfer = rng.uniform(0.38, 0.42, size=2)
    commands = (
        (1.0, *low),
        (1.0, transfer[0], low[1], low[2]),
        (1.0, transfer[0], transfer[1], low[2]),
    )
    return forward_preroll(
        model,
        command=commands[tank],
        reached=lambda state: float(state[tank]) >= target,
        control_dt=1.0,
        maximum_steps=500,
    )


def sample_training_episode(
    model, rng, reward_id, boundary: bool
) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(f"unsupported Three-Tank training reward {reward_id!r}")
    defaults = model.default_disturbances()
    start, target = _sample_tracking_equilibria(model, rng)
    if boundary:
        boundary_case = _sample_boundary_preroll(model, rng)
        episode = EpisodeSpec(
            initial_state=boundary_case["state"],
            initial_action=boundary_case["action"],
            reference=tuple(target["y_sp"]),
            horizon=_TRACKING_HORIZON,
            disturbances=defaults,
        )
    else:
        episode = EpisodeSpec(
            initial_state=tuple(start["state"]),
            initial_action=tuple(start["action"]),
            reference=tuple(target["y_sp"]),
            horizon=_TRACKING_HORIZON,
            disturbances=defaults,
        )
    return episode, "boundary-prerun" if boundary else "interior"


def sample_training_disturbance(model, rng):
    del model
    return _sample_bypass_schedule(rng)


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
        horizon=_TRACKING_HORIZON,
        reward_id="regulation",
        episode_factory=_disturbance_episode,
        metric_function=disturbance_rejection_episode_metrics,
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

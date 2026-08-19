"""Episode factories and formal benchmarks for the Three-Tank scenario."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from .model import NOMINAL_FLOW_M3S
from .metrics import (
    regulation_episode_metrics,
    thermal_regulation_episode_metrics,
)


_TRACKING_LEVEL_RANGE_M = (0.125, 0.4)
_TRACKING_TANK3_TEMPERATURE_RANGE_C = (20.5, 26.0)
_TRACKING_FLOW_RANGE_M3S = (1.0 / 60000.0, 6.0 / 60000.0)
_TRACKING_ACTION_RANGE = (0.02, 0.85)
_MINIMUM_LEVEL_MOVE_M = 0.05
_BOUNDARY_INITIAL_PROBABILITY = 0.20
_BOUNDARY_LEVEL_RANGE_FRACTION = (0.80, 0.92)
_MAXIMUM_SAMPLING_ATTEMPTS = 100


def make_default_episode(model) -> EpisodeSpec:
    start = model.nominal_steady_state()
    if not start["feasible"]:
        raise ValueError(
            "Three-Tank default equilibrium is infeasible for the resolved "
            f"parameters: {start['infeasible_reasons']}"
        )
    level_target = _feasible_tracking_equilibrium(
        model,
        level=0.375,
        temperature=float(start["y_sp"][5]),
        flow=NOMINAL_FLOW_M3S,
    )
    return EpisodeSpec(
        initial_state=tuple(start["state"]),
        initial_action=tuple(start["action"]),
        reference=tuple(start["y_sp"]),
        horizon=2400,
        disturbances=model.default_disturbances(),
        reference_schedule={
            600: tuple(level_target["y_sp"]),
        },
    )


def _feasible_tracking_equilibrium(model, *, level, temperature, flow):
    result = model.nominal_steady_state_for_tank3(
        tank_3_level=level,
        tank_3_temperature=temperature,
        flow=flow,
    )
    if not result["feasible"]:
        raise ValueError(
            "Three-Tank benchmark equilibrium is infeasible for the resolved "
            f"parameters: {result['infeasible_reasons']}"
        )
    return result


def _sample_tracking_equilibria(model, rng):
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        flow = float(rng.uniform(*_TRACKING_FLOW_RANGE_M3S))
        start_levels = rng.uniform(*_TRACKING_LEVEL_RANGE_M, size=3)
        target_levels = rng.uniform(*_TRACKING_LEVEL_RANGE_M, size=3)
        if np.any(np.abs(target_levels - start_levels) < _MINIMUM_LEVEL_MOVE_M):
            continue
        temperature = float(rng.uniform(*_TRACKING_TANK3_TEMPERATURE_RANGE_C))
        start = _sampled_tracking_equilibrium(
            model,
            levels=start_levels,
            temperature=temperature,
            flow=flow,
        )
        level_target = _sampled_tracking_equilibrium(
            model,
            levels=target_levels,
            temperature=temperature,
            flow=flow,
        )
        if start is not None and level_target is not None:
            return start, level_target
    raise ValueError(
        "could not sample feasible Three-Tank tracking equilibria within "
        f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _sampled_tracking_equilibrium(model, *, levels, temperature, flow):
    result = model.nominal_steady_state_for_tank3(
        tank_3_level=float(levels[2]),
        tank_3_temperature=temperature,
        flow=flow,
        upstream_levels=levels[:2],
    )
    action = np.asarray(result["action"], dtype=float)
    lower, upper = _TRACKING_ACTION_RANGE
    if (
        not result["feasible"]
        or np.any(action < lower)
        or np.any(action > upper)
    ):
        return None
    return result


def _tracking_episode(model, rng) -> EpisodeSpec:
    start, level_target = _sample_tracking_equilibria(model, rng)
    return EpisodeSpec(
        initial_state=tuple(start["state"]),
        initial_action=tuple(start["action"]),
        reference=tuple(start["y_sp"]),
        horizon=2400,
        disturbances=model.default_disturbances(),
        reference_schedule={
            600: tuple(level_target["y_sp"]),
        },
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    del rng
    steady = model.nominal_steady_state()
    if not steady["feasible"]:
        raise ValueError(
            "Three-Tank disturbance benchmark equilibrium is infeasible for "
            f"the resolved parameters: {steady['infeasible_reasons']}"
        )
    return EpisodeSpec(
        initial_state=tuple(steady["state"]),
        initial_action=tuple(steady["action"]),
        reference=tuple(steady["y_sp"]),
        horizon=3600,
        disturbances=model.default_disturbances(),
        disturbance_schedule={
            700: {
                "pump_flow_factor": 0.50,
                "v23_flow_factor": 0.50,
            },
            1200: {
                "pump_flow_factor": 1.0,
                "v23_flow_factor": 1.0,
            },
            1700: {"heat_loss_factor": 3.0},
            2300: {"heat_loss_factor": 1.0},
        },
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    del rng
    maximum = np.asarray(model.parameter("height_max"), dtype=float)
    temperatures = np.asarray(model.initial_state(), dtype=float)[1::2]
    levels = 0.9 * maximum
    initial = tuple(
        value
        for pair in zip(levels.tolist(), temperatures.tolist())
        for value in pair
    ) + (float(model.initial_state()[6]),)
    return EpisodeSpec(
        initial_state=initial,
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=1200,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(model, rng, reward_id) -> tuple[EpisodeSpec, str]:
    if reward_id not in {"regulation", "thermal_regulation"}:
        raise ValueError(f"unsupported Three-Tank training reward {reward_id!r}")
    defaults = model.default_disturbances()
    boundary_initial = rng.random() < _BOUNDARY_INITIAL_PROBABILITY
    start, target = _sample_tracking_equilibria(model, rng)
    if boundary_initial:
        maximum = np.asarray(model.parameter("height_max"), dtype=float)
        levels = rng.uniform(*_BOUNDARY_LEVEL_RANGE_FRACTION, size=3) * maximum
        initial = np.asarray(target["state"], dtype=float)
        initial[0::2][:3] = levels
        episode = EpisodeSpec(
            initial_state=tuple(initial),
            initial_action=tuple(target["action"]),
            reference=tuple(target["y_sp"]),
            horizon=2400,
            disturbances=defaults,
        )
    else:
        step = int(rng.integers(700, 1301))
        episode = EpisodeSpec(
            initial_state=tuple(start["state"]),
            initial_action=tuple(start["action"]),
            reference=tuple(start["y_sp"]),
            horizon=2400,
            disturbances=defaults,
            reference_schedule={step: tuple(target["y_sp"])},
        )
    return episode, "tracking"


def sample_training_disturbance(model, rng):
    del model
    start = int(rng.integers(450, 801))
    duration = int(rng.integers(300, 601))
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
        reward_id="thermal_regulation",
        episode_factory=_disturbance_episode,
        metric_function=thermal_regulation_episode_metrics,
        ranking_metrics=(
            ("unsafe_rate", "minimize"),
            ("return", "maximize"),
        ),
    ),
    "boundary-safety": Benchmark(
        id="boundary-safety",
        reward_id="thermal_regulation",
        episode_factory=_boundary_episode,
        metric_function=thermal_regulation_episode_metrics,
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

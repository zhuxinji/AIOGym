"""Episode factories and fixed Benchmarks for the heated Cascade scenario."""

from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._boundary import forward_preroll

from .metrics import regulation_episode_metrics


_LEVEL_RANGE_M = (0.125, 0.40)
_FLOW_RANGE_M3S = (2.0 / 60000.0, 6.0 / 60000.0)
_HEATER_DUTY_RANGE = (0.05, 0.80)
_MINIMUM_LEVEL_MOVE_M = 0.05
_MINIMUM_TEMPERATURE_MOVE_C = 1.0
_TEMPERATURE_MOVE_RANGE_C = (
    (-4.5, 6.5),
    (-3.5, 5.0),
    (-3.0, 3.5),
)
_ACTION_RANGE = (0.02, 0.85)
_BOUNDARY_HEATER_DUTY_RANGE = (0.80, 0.95)
_BOUNDARY_LEVEL_RANGE_FRACTION = (0.82, 0.848)
_MAXIMUM_SAMPLING_ATTEMPTS = 200
_DEFAULT_HORIZON = 600
_TRACKING_HORIZON = 2100
_BOUNDARY_HORIZON = 600
_TRAINING_HORIZON = 600

_DISTURBANCE_HORIZON = 1000


def make_default_episode(model) -> EpisodeSpec:
    start = model.nominal_steady_state()
    target_levels = [0.30, 0.25, 0.325]
    target_temperatures = model.steady_temperature_profile(
        heater_duties=(0.35, 0.35, 0.35)
    )
    target = model.nominal_steady_state(
        levels=target_levels,
        temperatures=target_temperatures,
        reservoir_volume=_target_reservoir_volume(model, start, target_levels),
    )
    if not start["feasible"] or not target["feasible"]:
        raise ValueError("default cascade tracking episode is infeasible")
    return EpisodeSpec(
        initial_state=tuple(start["state"]),
        initial_action=tuple(start["action"]),
        reference=tuple(target["y_sp"]),
        horizon=_DEFAULT_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _temperature_profile(model, rng, flow):
    heater_duties = rng.uniform(*_HEATER_DUTY_RANGE, size=3)
    return np.asarray(
        model.steady_temperature_profile(
            flow=flow,
            heater_duties=heater_duties,
        ),
        dtype=float,
    )


def _sample_temperature_pair(model, rng, flow):
    influenced = np.full(3, any(model.heater), dtype=bool)
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        start = _temperature_profile(model, rng, flow)
        target = _temperature_profile(model, rng, flow)
        signed_move = target - start
        move = np.abs(signed_move)
        move_range = np.asarray(_TEMPERATURE_MOVE_RANGE_C, dtype=float)
        if (
            np.all(move[influenced] >= _MINIMUM_TEMPERATURE_MOVE_C)
            and np.all(signed_move[influenced] >= move_range[influenced, 0])
            and np.all(signed_move[influenced] <= move_range[influenced, 1])
            and np.all(move[~influenced] <= 1e-10)
        ):
            return start, target
    raise ValueError(
        "could not sample distinct reachable Cascade temperature equilibria "
        f"within {_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


def _moved_levels(rng, start_levels):
    target = np.empty(3, dtype=float)
    for index, start_level in enumerate(start_levels):
        directions = []
        downward_limit = start_level - _LEVEL_RANGE_M[0]
        upward_limit = _LEVEL_RANGE_M[1] - start_level
        if downward_limit >= _MINIMUM_LEVEL_MOVE_M:
            directions.append((-1.0, downward_limit))
        if upward_limit >= _MINIMUM_LEVEL_MOVE_M:
            directions.append((1.0, upward_limit))
        direction, limit = directions[int(rng.integers(len(directions)))]
        target[index] = start_level + direction * float(
            rng.uniform(_MINIMUM_LEVEL_MOVE_M, limit)
        )
    return target


def _sample_tracking_equilibria(model, rng):
    for _attempt in range(_MAXIMUM_SAMPLING_ATTEMPTS):
        flow = float(rng.uniform(*_FLOW_RANGE_M3S))
        start_levels = rng.uniform(*_LEVEL_RANGE_M, size=3)
        target_levels = _moved_levels(rng, start_levels)
        start_temperatures, target_temperatures = _sample_temperature_pair(
            model, rng, flow
        )
        start = model.nominal_steady_state(
            flow=flow,
            levels=start_levels,
            temperatures=start_temperatures,
        )
        target = model.nominal_steady_state(
            flow=flow,
            levels=target_levels,
            temperatures=target_temperatures,
            reservoir_volume=_target_reservoir_volume(
                model,
                start,
                target_levels,
            ),
        )
        lower, upper = _ACTION_RANGE
        if all(
            point["feasible"]
            and _action_matches_heater_configuration(
                model,
                point["action"],
                lower=lower,
                upper=upper,
            )
            for point in (start, target)
        ):
            return start, target
    raise ValueError(
        "could not sample feasible cascade tracking equilibria within "
        f"{_MAXIMUM_SAMPLING_ATTEMPTS} attempts"
    )


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
    _initialize_reservoir_at_ambient(model, steady)
    if not steady["feasible"]:
        raise ValueError("default cascade disturbance operating point is infeasible")
    defaults = model.default_disturbances()
    start = int(rng.integers(300, 401))
    duration = int(rng.integers(200, 301))
    event = {
        "pump_flow_factor": float(rng.uniform(0.72, 0.90)),
        "ambient_temperature": float(rng.uniform(16.0, 19.0)),
        **_heater_efficiency_shift(model, rng, 0.55, 0.80),
    }
    return EpisodeSpec(
        initial_state=tuple(steady["state"]),
        initial_action=tuple(steady["action"]),
        reference=tuple(steady["y_sp"]),
        horizon=_DISTURBANCE_HORIZON,
        disturbances=defaults,
        disturbance_schedule={
            start: event,
            start + duration: defaults,
        },
    )


def _sample_boundary_preroll(model, rng):
    maximum = np.asarray(model.parameter("height_max"), dtype=float)
    tank = int(rng.integers(3))
    target = float(rng.uniform(*_BOUNDARY_LEVEL_RANGE_FRACTION)) * maximum[tank]
    hydraulic_commands = (
        (1.0, 0.05, 0.05, 0.05),
        (1.0, 0.40, 0.05, 0.05),
        (1.0, 0.40, 0.40, 0.05),
    )
    command = (
        *hydraulic_commands[tank],
        *(
            rng.uniform(*_BOUNDARY_HEATER_DUTY_RANGE, size=3)
            * np.asarray(model.heater, dtype=float)
        ),
    )
    return forward_preroll(
        model,
        command=command,
        reached=lambda state: float(state[2 * tank]) >= target,
        control_dt=2.0,
        maximum_steps=250,
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    boundary = _sample_boundary_preroll(model, rng)
    return EpisodeSpec(
        initial_state=tuple(boundary["state"]),
        initial_action=tuple(boundary["action"]),
        reference=tuple(model.default_setpoint_vector()),
        horizon=_BOUNDARY_HORIZON,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(
    model, rng, reward_id, boundary: bool
) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(f"unsupported cascade training reward {reward_id!r}")
    start, target = _sample_tracking_equilibria(model, rng)
    if boundary:
        boundary_case = _sample_boundary_preroll(model, rng)
        initial_state = tuple(boundary_case["state"])
        initial_action = tuple(boundary_case["action"])
        initial_family = "boundary-prerun"
    else:
        initial_state = tuple(start["state"])
        initial_action = tuple(start["action"])
        initial_family = "interior"
    return (
        EpisodeSpec(
            initial_state=initial_state,
            initial_action=initial_action,
            reference=tuple(target["y_sp"]),
            horizon=_TRAINING_HORIZON,
            disturbances=model.default_disturbances(),
        ),
        initial_family,
    )


def sample_training_disturbance(model, rng):
    defaults = model.default_disturbances()
    start = int(rng.integers(125, 201))
    duration = int(rng.integers(100, 176))
    return {
        start: {
            "pump_flow_factor": float(rng.uniform(0.75, 0.92)),
            "ambient_temperature": float(rng.uniform(16.0, 19.0)),
            **_heater_efficiency_shift(model, rng, 0.60, 0.85),
        },
        start + duration: defaults,
    }


def _action_matches_heater_configuration(model, action, *, lower, upper):
    values = np.asarray(action, dtype=float)
    hydraulic = values[:4]
    heaters = values[4:]
    available = np.asarray(model.heater, dtype=bool)
    return bool(
        np.all((lower <= hydraulic) & (hydraulic <= upper))
        and np.all((lower <= heaters[available]) & (heaters[available] <= upper))
        and np.allclose(heaters[~available], 0.0, rtol=0.0, atol=1e-12)
    )


def _target_reservoir_volume(model, start, target_levels):
    area = np.asarray(model.parameter("area"), dtype=float)
    start_levels = np.asarray(start["y_sp"][:3], dtype=float)
    target = np.asarray(target_levels, dtype=float)
    return float(
        start["reservoir_volume_m3"]
        + np.sum(area * (start_levels - target))
    )


def _initialize_reservoir_at_ambient(model, point):
    point["state"][7] = float(model.parameter("ambient_temperature"))


def _heater_efficiency_shift(model, rng, lower, upper):
    return {
        f"heater_H{index + 1}_efficiency_factor": float(rng.uniform(lower, upper))
        for index, enabled in enumerate(model.heater)
        if enabled
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
            ("settling_rate", "maximize"),
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
            ("settling_rate", "maximize"),
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
            ("settling_rate", "maximize"),
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

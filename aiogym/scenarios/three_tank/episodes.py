"""Episode factories and formal benchmarks for the Three-Tank scenario."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._metrics import regulation_episode_metrics


_MEASUREMENT_NOISE = {"std": 0.001, "bias_std": 0.0}


def make_default_episode(model) -> EpisodeSpec:
    start = model.nominal_steady_state()
    if not start["feasible"]:
        raise ValueError(
            "Three-Tank default equilibrium is infeasible for the resolved "
            f"parameters: {start['infeasible_reasons']}"
        )
    target = _feasible_tracking_equilibrium(
        model,
        level=0.30,
        temperature=23.7,
        flow=5.0 / 60000.0,
    )
    return EpisodeSpec(
        initial_state=tuple(start["state"]),
        reference=tuple(start["y_sp"]),
        horizon=3000,
        disturbances=model.default_disturbances(),
        reference_schedule={600: tuple(target["y_sp"])},
    )


def _feasible_tracking_equilibrium(model, *, level, temperature, flow):
    result = model.nominal_steady_state_for_tank3(
        tank_3_level=level,
        tank_3_temperature=temperature,
        flow=flow,
        upstream_levels=(level, level),
    )
    if not result["feasible"]:
        raise ValueError(
            "Three-Tank benchmark equilibrium is infeasible for the resolved "
            f"parameters: {result['infeasible_reasons']}"
        )
    return result


def _tracking_episode(model) -> EpisodeSpec:
    start = _feasible_tracking_equilibrium(
        model,
        level=0.12,
        temperature=20.5,
        flow=5.0 / 60000.0,
    )
    target = _feasible_tracking_equilibrium(
        model,
        level=0.30,
        temperature=23.0,
        flow=5.0 / 60000.0,
    )
    return EpisodeSpec(
        initial_state=tuple(start["state"]),
        reference=tuple(start["y_sp"]),
        horizon=4200,
        disturbances=model.default_disturbances(),
        reference_schedule={900: tuple(target["y_sp"])},
    )


def _disturbance_episode(model) -> EpisodeSpec:
    steady = model.nominal_steady_state()
    if not steady["feasible"]:
        raise ValueError(
            "Three-Tank disturbance benchmark equilibrium is infeasible for "
            f"the resolved parameters: {steady['infeasible_reasons']}"
        )
    return EpisodeSpec(
        initial_state=tuple(steady["state"]),
        reference=tuple(steady["y_sp"]),
        horizon=2400,
        disturbances=model.default_disturbances(),
        disturbance_schedule={
            800: {"pump_flow_factor": 0.85},
            1400: {"pump_flow_factor": 1.0},
        },
    )


def _boundary_episode(model) -> EpisodeSpec:
    maximum = np.asarray(model.parameter("height_max"), dtype=float)
    temperatures = np.asarray(model.initial_state(), dtype=float)[1::2]
    levels = 0.9 * maximum
    initial = tuple(
        value
        for pair in zip(levels.tolist(), temperatures.tolist())
        for value in pair
    )
    return EpisodeSpec(
        initial_state=initial,
        reference=tuple(model.default_setpoint_vector()),
        horizon=1200,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(model, rng) -> tuple[EpisodeSpec, str]:
    family = (
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    )[int(rng.integers(0, 3))]
    defaults = model.default_disturbances()
    if family == "tracking":
        maximum = float(model.parameter("height_max")[2])
        for _attempt in range(100):
            levels = rng.uniform(0.25, 0.75, size=2) * maximum
            if abs(levels[1] - levels[0]) < 0.25 * maximum:
                continue
            temperatures = rng.uniform(20.5, 26.0, size=2)
            if abs(temperatures[1] - temperatures[0]) < 2.0:
                continue
            flows_lpm = np.empty(2, dtype=float)
            for index, temperature in enumerate(temperatures):
                upper = min(12.0, 24.0 / max(float(temperature) - 20.0, 0.5))
                flows_lpm[index] = rng.uniform(2.0, upper)
            flows = flows_lpm / 60000.0
            start = model.nominal_steady_state_for_tank3(
                tank_3_level=float(levels[0]),
                tank_3_temperature=float(temperatures[0]),
                flow=float(flows[0]),
                upstream_levels=(float(levels[0]),) * 2,
            )
            target = model.nominal_steady_state_for_tank3(
                tank_3_level=float(levels[1]),
                tank_3_temperature=float(temperatures[1]),
                flow=float(flows[1]),
                upstream_levels=(float(levels[1]),) * 2,
            )
            if start["feasible"] and target["feasible"]:
                break
        else:
            raise ValueError(
                "could not sample a feasible Three-Tank tracking episode"
            )
        step = int(rng.integers(700, 1301))
        episode = EpisodeSpec(
            initial_state=tuple(start["state"]),
            reference=tuple(start["y_sp"]),
            horizon=2400,
            disturbances=defaults,
            reference_schedule={step: tuple(target["y_sp"])},
        )
    elif family == "disturbance-rejection":
        steady = model.nominal_steady_state()
        if not steady["feasible"]:
            raise ValueError(
                "could not sample a feasible Three-Tank disturbance episode"
            )
        start = int(rng.integers(450, 801))
        duration = int(rng.integers(300, 601))
        factor = float(rng.uniform(0.72, 0.94))
        episode = EpisodeSpec(
            initial_state=tuple(steady["state"]),
            reference=tuple(steady["y_sp"]),
            horizon=1800,
            disturbances=defaults,
            disturbance_schedule={
                start: {"pump_flow_factor": factor},
                start + duration: {"pump_flow_factor": 1.0},
            },
        )
    else:
        maximum = np.asarray(model.parameter("height_max"), dtype=float)
        levels = rng.uniform(0.80, 0.92, size=3) * maximum
        temperatures = np.asarray(model.initial_state(), dtype=float)[1::2]
        temperatures = temperatures + rng.uniform(0.0, 4.0, size=3)
        initial = tuple(
            value
            for pair in zip(levels.tolist(), temperatures.tolist())
            for value in pair
        )
        episode = EpisodeSpec(
            initial_state=initial,
            reference=tuple(model.default_setpoint_vector()),
            horizon=1200,
            disturbances=defaults,
        )
    return episode, family


BENCHMARKS = {
    "tracking": Benchmark(
        id="tracking",
        make_episode=_tracking_episode,
        metric_function=regulation_episode_metrics,
        ranking_metrics=(
            ("unsafe_rate", "minimize"),
            ("tracking_iae", "minimize"),
        ),
        measurement_noise=_MEASUREMENT_NOISE,
    ),
    "disturbance-rejection": Benchmark(
        id="disturbance-rejection",
        make_episode=_disturbance_episode,
        metric_function=regulation_episode_metrics,
        ranking_metrics=(
            ("unsafe_rate", "minimize"),
            ("disturbance_iae", "minimize"),
            ("recovery_time", "minimize"),
        ),
        measurement_noise=_MEASUREMENT_NOISE,
    ),
    "boundary-safety": Benchmark(
        id="boundary-safety",
        make_episode=_boundary_episode,
        metric_function=regulation_episode_metrics,
        ranking_metrics=(
            ("unsafe_rate", "minimize"),
            ("time_to_violation", "maximize"),
            ("tracking_iae", "minimize"),
        ),
        measurement_noise=_MEASUREMENT_NOISE,
    ),
}


__all__ = ["BENCHMARKS", "make_default_episode", "sample_training_episode"]

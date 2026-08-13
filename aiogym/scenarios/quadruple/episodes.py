"""Episode factories and formal benchmarks for the Quadruple-Tank scenario."""
from __future__ import annotations

import numpy as np

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._metrics import regulation_episode_metrics


_MEASUREMENT_NOISE = {"std": 0.001, "bias_std": 0.0}


def make_default_episode(model) -> EpisodeSpec:
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=600,
        disturbances=model.default_disturbances(),
        reference_schedule={300: (15.0, 10.5)},
    )


def _tracking_episode(model) -> EpisodeSpec:
    reference = np.asarray(model.default_setpoint_vector(), dtype=float)
    maximum = float(model.parameter("max_level"))
    first = maximum * np.asarray((0.8, 0.5), dtype=float)
    second = maximum * np.asarray((0.5, 0.8), dtype=float)
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        reference=tuple(reference),
        horizon=600,
        disturbances=model.default_disturbances(),
        reference_schedule={120: tuple(first), 360: tuple(second)},
    )


def _disturbance_episode(model) -> EpisodeSpec:
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=600,
        disturbances=model.default_disturbances(),
        disturbance_schedule={
            150: {"pump_flow_factor": 0.85},
            350: {"pump_flow_factor": 1.0},
        },
    )


def _boundary_episode(model) -> EpisodeSpec:
    maximum = float(model.parameter("max_level"))
    initial = (0.9 * maximum,) * 4
    reference = np.clip(
        np.asarray(model.default_setpoint_vector(), dtype=float), 0.0, maximum
    )
    return EpisodeSpec(
        initial_state=initial,
        reference=tuple(reference),
        horizon=400,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(model, rng) -> tuple[EpisodeSpec, str]:
    family = (
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    )[int(rng.integers(0, 3))]
    maximum = float(model.parameter("max_level"))
    reference = np.asarray(model.default_setpoint_vector(), dtype=float)
    if family == "tracking":
        step = int(rng.integers(90, 241))
        for _attempt in range(100):
            target = rng.uniform(0.35 * maximum, 0.80 * maximum, size=2)
            if (
                np.all(np.abs(target - reference) >= 0.15 * maximum)
                and model.tracking_steady_state_action(target) is not None
            ):
                break
        else:
            raise ValueError(
                "could not sample a feasible Quadruple-Tank tracking episode"
            )
        episode = EpisodeSpec(
            initial_state=tuple(model.initial_state()),
            reference=tuple(reference),
            horizon=600,
            disturbances=model.default_disturbances(),
            reference_schedule={step: tuple(target)},
        )
    elif family == "disturbance-rejection":
        start = int(rng.integers(120, 241))
        duration = int(rng.integers(120, 241))
        disturbance = float(rng.uniform(0.78, 0.94))
        episode = EpisodeSpec(
            initial_state=tuple(model.initial_state()),
            reference=tuple(reference),
            horizon=600,
            disturbances=model.default_disturbances(),
            disturbance_schedule={
                start: {"pump_flow_factor": disturbance},
                start + duration: {"pump_flow_factor": 1.0},
            },
        )
    else:
        initial = tuple(rng.uniform(0.82, 0.92, size=4) * maximum)
        episode = EpisodeSpec(
            initial_state=initial,
            reference=tuple(np.clip(reference, 0.0, maximum)),
            horizon=400,
            disturbances=model.default_disturbances(),
        )
    return episode, family


BENCHMARKS = {
    "tracking": Benchmark(
        id="tracking",
        make_episode=_tracking_episode,
        metric_function=regulation_episode_metrics,
        ranking_metrics=(("tracking_iae", "minimize"),),
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

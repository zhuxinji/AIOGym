"""Batch episodes and formal Benchmarks for crystallization."""
from __future__ import annotations

from aiogym.core.specs import Benchmark, EpisodeSpec
from aiogym.scenarios._metrics import regulation_episode_metrics


_BATCH_HORIZON = 100
_TRACKING_HORIZON = 50
_TARGET_ACTION_RANGE = (0.10, 0.90)
_TRACKING_TARGET_ACTION_RANGE = (0.80, 0.90)


def make_default_episode(model) -> EpisodeSpec:
    target_action = (0.85,)
    endpoint = model.batch_endpoint(
        target_action,
        horizon_steps=_TRACKING_HORIZON,
    )
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=target_action,
        reference=tuple(endpoint["output"]),
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _reachable_batch_target(
    model,
    rng,
    *,
    initial_state=None,
    horizon_steps=_BATCH_HORIZON,
    action_range=_TARGET_ACTION_RANGE,
):
    action = float(rng.uniform(*action_range))
    endpoint = model.batch_endpoint(
        [action],
        horizon_steps=horizon_steps,
        initial_state=initial_state,
    )
    return {
        "action": (action,),
        "reference": tuple(float(value) for value in endpoint["output"]),
    }


def _tracking_episode(model, rng) -> EpisodeSpec:
    target = _reachable_batch_target(
        model,
        rng,
        horizon_steps=_TRACKING_HORIZON,
        action_range=_TRACKING_TARGET_ACTION_RANGE,
    )
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=target["action"],
        reference=target["reference"],
        horizon=_TRACKING_HORIZON,
        disturbances=model.default_disturbances(),
    )


def _disturbance_episode(model, rng) -> EpisodeSpec:
    start = int(rng.integers(20, 36))
    duration = int(rng.integers(30, 51))
    return EpisodeSpec(
        initial_state=tuple(model.initial_state()),
        initial_action=tuple(model.default_action()),
        reference=tuple(model.default_setpoint_vector()),
        horizon=_BATCH_HORIZON,
        disturbances=model.default_disturbances(),
        disturbance_schedule={
            start: {
                "growth_rate_factor": float(rng.uniform(0.80, 0.95)),
                "nucleation_rate_factor": float(rng.uniform(1.05, 1.20)),
                "solubility_concentration_bias": float(rng.uniform(2.0, 6.0)),
            },
            start + duration: model.default_disturbances(),
        },
    )


def _boundary_episode(model, rng) -> EpisodeSpec:
    initial_state = tuple(
        [*model.initial_state()[:4], float(rng.uniform(1.70, 1.95))]
    )
    action = (float(rng.uniform(0.10, 0.50)),)
    endpoint = model.batch_endpoint(
        action,
        horizon_steps=_BATCH_HORIZON,
        initial_state=initial_state,
    )
    return EpisodeSpec(
        initial_state=initial_state,
        initial_action=action,
        reference=tuple(endpoint["output"]),
        horizon=_BATCH_HORIZON,
        disturbances=model.default_disturbances(),
    )


def sample_training_episode(
    model, rng, reward_id, boundary: bool
) -> tuple[EpisodeSpec, str]:
    if reward_id != "regulation":
        raise ValueError(
            f"unsupported crystallization training reward {reward_id!r}"
        )
    boundary_concentration = float(rng.uniform(1.70, 1.95))
    if boundary:
        initial_state = tuple(
            [*model.initial_state()[:4], boundary_concentration]
        )
    else:
        initial_state = tuple(model.initial_state())
    target = _reachable_batch_target(
        model,
        rng,
        initial_state=initial_state,
        horizon_steps=_TRACKING_HORIZON,
        action_range=_TRACKING_TARGET_ACTION_RANGE,
    )
    return (
        EpisodeSpec(
            initial_state=initial_state,
            initial_action=target["action"],
            reference=target["reference"],
            horizon=_TRACKING_HORIZON,
            disturbances=model.default_disturbances(),
        ),
        "boundary-batch" if boundary else "interior",
    )


def sample_training_disturbance(model, rng):
    start = int(rng.integers(10, 18))
    duration = int(rng.integers(15, 26))
    return {
        start: {
            "growth_rate_factor": float(rng.uniform(0.80, 1.20)),
            "nucleation_rate_factor": float(rng.uniform(0.80, 1.20)),
            "solubility_concentration_bias": float(rng.uniform(-5.0, 5.0)),
        },
        start + duration: model.default_disturbances(),
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

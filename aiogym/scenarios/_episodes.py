"""Shared assembly for equilibrium-to-equilibrium tracking episodes."""
from aiogym.core.specs import EpisodeSpec


def tracking_episode(model, rng, *, sample_equilibrium, horizon):
    start = sample_equilibrium(model, rng)
    target = sample_equilibrium(model, rng, previous_reference=start["reference"])
    return EpisodeSpec(
        initial_state=start["state"],
        initial_action=start["action"],
        reference=target["reference"],
        horizon=horizon,
        disturbances=model.default_disturbances(),
    )

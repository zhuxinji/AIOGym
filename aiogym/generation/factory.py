"""Resolve registered DistributionSpecs into runtime samplers."""
from __future__ import annotations

from .registry import load_distribution, sampler_factory_for
from .specs import DistributionSpec


def make_episode_sampler(
    distribution: DistributionSpec | str,
    *,
    split: str = "training",
):
    resolved = (
        distribution
        if isinstance(distribution, DistributionSpec)
        else load_distribution(distribution)
    )
    factory = sampler_factory_for(resolved.distribution_id)
    sampler = factory(resolved, split)
    if sampler.distribution_id != resolved.distribution_id:
        raise RuntimeError("episode sampler distribution ID mismatch")
    if sampler.distribution_hash != resolved.distribution_hash:
        raise RuntimeError("episode sampler distribution hash mismatch")
    return sampler


__all__ = ["make_episode_sampler"]

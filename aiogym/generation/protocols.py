"""Internal protocols for deterministic episode generation."""
from __future__ import annotations

from typing import Protocol

from .specs import EpisodeSpec


class EpisodeSampler(Protocol):
    distribution_id: str
    distribution_hash: str

    def sample(
        self,
        base_seed: int,
        *,
        episode_index: int,
    ) -> EpisodeSpec: ...


__all__ = ["EpisodeSampler"]

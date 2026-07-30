"""Deterministic Phase-A episode resolvers."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from aiogym.models.cases import load_case

from .adapters import distribution_spec_from_case
from .seed_tree import SeedTree
from .specs import DistributionSpec, EpisodeSpec


class FixedCaseEpisodeSampler:
    """Resolve a fixed Case v2 profile into immutable EpisodeSpecs."""

    def __init__(
        self,
        case: str | Mapping[str, Any],
        *,
        scenario: str | None = None,
        goal: str = "regulation",
        split: str = "validation",
    ) -> None:
        self.case_profile = load_case(case, scenario=scenario)
        self.distribution = distribution_spec_from_case(
            self.case_profile,
            goal=goal,
        )
        self.split = str(split)

    @property
    def distribution_id(self) -> str:
        return self.distribution.distribution_id

    @property
    def distribution_hash(self) -> str:
        return self.distribution.distribution_hash

    def sample(
        self,
        seed: int,
        *,
        worker_index: int = 0,
        episode_index: int = 0,
    ) -> EpisodeSpec:
        """Return the same fully resolved episode for the same public identity."""

        tree = SeedTree.for_split(
            seed,
            self.distribution.distribution_id,
            self.split,
            worker_index=worker_index,
            episode_index=episode_index,
        )
        return _fixed_distribution_episode(self.distribution, tree)


def episode_spec_from_case(
    case: str | Mapping[str, Any],
    *,
    seed: int,
    scenario: str | None = None,
    goal: str = "regulation",
    split: str = "validation",
    worker_index: int = 0,
    episode_index: int = 0,
) -> EpisodeSpec:
    """Convenience wrapper for deterministic fixed-case resolution."""

    return FixedCaseEpisodeSampler(
        case,
        scenario=scenario,
        goal=goal,
        split=split,
    ).sample(
        seed,
        worker_index=worker_index,
        episode_index=episode_index,
    )


def _fixed_distribution_episode(
    distribution: DistributionSpec,
    seed_tree: SeedTree,
) -> EpisodeSpec:
    declaration = distribution.declaration
    plant = declaration["plant_distribution"]
    initial = declaration["initial_state_distribution"]
    reference = declaration["reference_distribution"]
    disturbance = declaration["disturbance_distribution"]
    economic = declaration.get("economic_context_distribution") or {}
    for name, section in (
        ("plant_distribution", plant),
        ("initial_state_distribution", initial),
        ("reference_distribution", reference),
        ("disturbance_distribution", disturbance),
    ):
        if section.get("kind") != "fixed":
            raise ValueError(
                f"{name} must have kind='fixed' for fixed-case sampling"
            )
    status = str(
        plant.get("case_id", distribution.distribution_id)
    )
    return EpisodeSpec.from_distribution(
        distribution,
        base_seed=seed_tree.base_seed,
        component_seeds=seed_tree.component_seeds,
        plant_parameters=deepcopy(plant["parameters"]),
        initial_state=deepcopy(initial["state"]),
        reference_schedule=deepcopy(reference["schedule"]),
        disturbance_schedule=deepcopy(disturbance["schedule"]),
        sensor_model=deepcopy(declaration["sensor_distribution"]),
        actuator_model=deepcopy(declaration["actuator_distribution"]),
        economic_context=deepcopy(economic.get("context", {})),
        difficulty_tags=("fixed-case", status),
    )


__all__ = [
    "FixedCaseEpisodeSampler",
    "episode_spec_from_case",
]

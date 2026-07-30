"""Versioned episode-generation contracts.

This package separates immutable descriptions and resolved episode inputs from
the runtime environments that consume them.
"""
from .adapters import distribution_spec_from_case
from .curriculum import (
    CURRICULUM_LEVELS,
    CURRICULUM_SCHEMA_VERSION,
    QUADRUPLE_CURRICULUM_V1,
    CurriculumLevel,
    CurriculumSpec,
    get_curriculum_level,
)
from .cascade import (
    CASCADE_PRODUCT_FLOW_M3S,
    CascadeTrainingSampler,
    cascade_training_distribution,
    validate_cascade_episode,
)
from .cascade_recirculating import (
    CascadeRecirculatingTrainingSampler,
    cascade_recirculating_training_distribution,
    validate_cascade_recirculating_episode,
)
from .feasibility import FeasibilitySamplingError
from .quadruple import (
    QuadrupleCurriculumSampler,
    QuadrupleDisturbanceGenerator,
    QuadrupleReferenceGenerator,
    QuadrupleTrainingSampler,
    quadruple_training_distribution,
    validate_quadruple_episode,
    validate_quadruple_parameters,
)
from .factory import make_episode_sampler
from .protocols import EpisodeSampler
from .registry import (
    list_distributions,
    load_distribution,
    register_distribution,
)
from .samplers import FixedCaseEpisodeSampler, episode_spec_from_case
from .seed_tree import (
    SEED_COMPONENTS,
    SEED_SPLITS,
    SeedTree,
    seed_namespace,
)
from .specs import (
    DISTRIBUTION_SCHEMA_VERSION,
    EPISODE_SPEC_SCHEMA_VERSION,
    DistributionSpec,
    EpisodeSpec,
)
from .validation import validate_episode_for_env


__all__ = [
    "DISTRIBUTION_SCHEMA_VERSION",
    "EPISODE_SPEC_SCHEMA_VERSION",
    "CURRICULUM_LEVELS",
    "CURRICULUM_SCHEMA_VERSION",
    "QUADRUPLE_CURRICULUM_V1",
    "SEED_COMPONENTS",
    "SEED_SPLITS",
    "DistributionSpec",
    "EpisodeSpec",
    "EpisodeSampler",
    "FixedCaseEpisodeSampler",
    "CurriculumLevel",
    "CurriculumSpec",
    "CASCADE_PRODUCT_FLOW_M3S",
    "CascadeTrainingSampler",
    "CascadeRecirculatingTrainingSampler",
    "FeasibilitySamplingError",
    "QuadrupleCurriculumSampler",
    "QuadrupleDisturbanceGenerator",
    "QuadrupleReferenceGenerator",
    "QuadrupleTrainingSampler",
    "SeedTree",
    "cascade_training_distribution",
    "cascade_recirculating_training_distribution",
    "distribution_spec_from_case",
    "episode_spec_from_case",
    "get_curriculum_level",
    "list_distributions",
    "load_distribution",
    "make_episode_sampler",
    "quadruple_training_distribution",
    "register_distribution",
    "seed_namespace",
    "validate_quadruple_episode",
    "validate_quadruple_parameters",
    "validate_episode_for_env",
    "validate_cascade_episode",
    "validate_cascade_recirculating_episode",
]

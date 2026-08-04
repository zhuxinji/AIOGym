"""Usable constrained, recurrent, and safety-aware research components.

This module is intentionally outside the v1 stable compatibility promise and
may change in a minor release. It must be imported explicitly as
``aiogym.experimental.rl``.
"""

from aiogym.rl.constrained import (
    LAGRANGIAN_SAC_STATE_SCHEMA_VERSION,
    ConstrainedReplayBuffer,
    LagrangeMultiplier,
    LagrangianSAC,
)
from aiogym.rl.observations import (
    OBSERVATION_CONTRACT_SCHEMA_VERSION,
    HistoryObservationWrapper,
    ObservationContract,
    RecurrentStateContract,
    wrap_observation_contract,
)
from aiogym.rl.safety import (
    SAFETY_SHIELD_SCHEMA_VERSION,
    ProjectionSafetyShield,
    SafetyShieldWrapper,
    ShieldDecision,
)


__all__ = [
    "LAGRANGIAN_SAC_STATE_SCHEMA_VERSION",
    "OBSERVATION_CONTRACT_SCHEMA_VERSION",
    "SAFETY_SHIELD_SCHEMA_VERSION",
    "ConstrainedReplayBuffer",
    "HistoryObservationWrapper",
    "LagrangeMultiplier",
    "LagrangianSAC",
    "ObservationContract",
    "ProjectionSafetyShield",
    "RecurrentStateContract",
    "SafetyShieldWrapper",
    "ShieldDecision",
    "wrap_observation_contract",
]

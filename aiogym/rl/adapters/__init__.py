"""Algorithm adapters used by the unified experiment runner."""

from .base import TrainingAdapter
from .bc import BCTrainingAdapter
from .rlpd import RLPDTrainingAdapter
from .sb3 import SB3TrainingAdapter


def build_training_adapter(plan):
    algorithm = plan.config.algorithm_id
    if algorithm in {"ppo", "sac", "td3"}:
        return SB3TrainingAdapter()
    if algorithm == "rlpd":
        return RLPDTrainingAdapter()
    if algorithm == "bc":
        return BCTrainingAdapter()
    raise ValueError(
        f"algorithm {algorithm!r} is experimental and has no stable adapter"
    )


__all__ = [
    "TrainingAdapter",
    "BCTrainingAdapter",
    "RLPDTrainingAdapter",
    "SB3TrainingAdapter",
    "build_training_adapter",
]

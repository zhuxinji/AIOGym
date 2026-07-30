"""Update-to-data accounting independent of vector-environment width."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass
class UTDController:
    """Accumulate optimizer update credit per collected transition."""

    ratio: float
    transitions: int = 0
    updates: int = 0

    def __post_init__(self) -> None:
        self.ratio = float(self.ratio)
        if not math.isfinite(self.ratio) or self.ratio < 0.0:
            raise ValueError("UTD ratio must be finite and non-negative")
        if self.transitions < 0 or self.updates < 0:
            raise ValueError("UTD counters must be non-negative")
        if self.updates > math.floor(self.transitions * self.ratio + 1e-12):
            raise ValueError("updates exceed accumulated UTD credit")

    def observe(self, transition_count: int) -> int:
        """Return updates now due after observing a transition batch."""

        if isinstance(transition_count, bool) or not isinstance(
            transition_count, int
        ):
            raise TypeError("transition_count must be an integer")
        if transition_count < 0:
            raise ValueError("transition_count must be non-negative")
        self.transitions += transition_count
        target = math.floor(self.transitions * self.ratio + 1e-12)
        due = target - self.updates
        self.updates = target
        return due

    def state_dict(self) -> dict[str, float | int]:
        return {
            "ratio": self.ratio,
            "transitions": self.transitions,
            "updates": self.updates,
        }

    @classmethod
    def from_state_dict(cls, state) -> "UTDController":
        return cls(
            ratio=state["ratio"],
            transitions=int(state["transitions"]),
            updates=int(state["updates"]),
        )


def sb3_update_schedule(
    *,
    utd_ratio: float,
    n_envs: int,
    vector_steps: int = 1,
) -> tuple[int, int]:
    """Translate explicit UTD into SB3 train_freq/gradient_steps."""

    if n_envs <= 0 or vector_steps <= 0:
        raise ValueError("n_envs and vector_steps must be positive")
    ratio = float(utd_ratio)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("utd_ratio must be finite and non-negative")
    if ratio == 0.0:
        return int(vector_steps), 0
    fraction = Fraction(str(ratio)).limit_denominator(10_000)
    required_multiple = fraction.denominator // math.gcd(
        fraction.denominator,
        n_envs,
    )
    scheduled_steps = (
        math.ceil(vector_steps / required_multiple) * required_multiple
    )
    gradient_steps = (
        fraction.numerator
        * n_envs
        * scheduled_steps
        // fraction.denominator
    )
    return int(scheduled_steps), int(gradient_steps)


__all__ = ["UTDController", "sb3_update_schedule"]

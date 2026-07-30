"""Versioned curriculum levels for programmatic episode generation."""
from __future__ import annotations

from dataclasses import dataclass


CURRICULUM_SCHEMA_VERSION = "aiogym.curriculum.v1"


@dataclass(frozen=True)
class CurriculumLevel:
    """One fixed and auditable difficulty configuration."""

    level_id: str
    plant_relative_std: float
    operating_action_half_width: float
    initial_relative_std: float
    reference_patterns: tuple[str, ...]
    reference_event_min: int
    reference_event_max: int
    reference_action_half_width: float
    minimum_dwell_steps: int
    disturbance_probability: float
    disturbance_patterns: tuple[str, ...]
    disturbance_max_shift: float
    sensor_noise_pct: float
    recovery_probability: float
    state_safety_margin: float

    def metadata(self) -> dict:
        return {
            "level_id": self.level_id,
            "plant_relative_std": self.plant_relative_std,
            "operating_action_half_width": (
                self.operating_action_half_width
            ),
            "initial_relative_std": self.initial_relative_std,
            "reference_patterns": list(self.reference_patterns),
            "reference_event_min": self.reference_event_min,
            "reference_event_max": self.reference_event_max,
            "reference_action_half_width": (
                self.reference_action_half_width
            ),
            "minimum_dwell_steps": self.minimum_dwell_steps,
            "disturbance_probability": self.disturbance_probability,
            "disturbance_patterns": list(self.disturbance_patterns),
            "disturbance_max_shift": self.disturbance_max_shift,
            "sensor_noise_pct": self.sensor_noise_pct,
            "recovery_probability": self.recovery_probability,
            "state_safety_margin": self.state_safety_margin,
        }


_LEVELS = (
    CurriculumLevel(
        level_id="L0",
        plant_relative_std=0.0,
        operating_action_half_width=0.015,
        initial_relative_std=0.01,
        reference_patterns=("single_step",),
        reference_event_min=1,
        reference_event_max=1,
        reference_action_half_width=0.025,
        minimum_dwell_steps=100,
        disturbance_probability=0.0,
        disturbance_patterns=("step",),
        disturbance_max_shift=0.05,
        sensor_noise_pct=0.0,
        recovery_probability=0.0,
        state_safety_margin=0.5,
    ),
    CurriculumLevel(
        level_id="L1",
        plant_relative_std=0.0,
        operating_action_half_width=0.04,
        initial_relative_std=0.04,
        reference_patterns=(
            "steady_hold",
            "single_step",
            "async_step",
            "multi_step",
            "ramp",
        ),
        reference_event_min=1,
        reference_event_max=3,
        reference_action_half_width=0.06,
        minimum_dwell_steps=80,
        disturbance_probability=0.0,
        disturbance_patterns=("step", "pulse"),
        disturbance_max_shift=0.08,
        sensor_noise_pct=0.0,
        recovery_probability=0.0,
        state_safety_margin=0.4,
    ),
    CurriculumLevel(
        level_id="L2",
        plant_relative_std=0.04,
        operating_action_half_width=0.055,
        initial_relative_std=0.06,
        reference_patterns=(
            "steady_hold",
            "single_step",
            "async_step",
            "multi_step",
            "ramp",
        ),
        reference_event_min=1,
        reference_event_max=3,
        reference_action_half_width=0.08,
        minimum_dwell_steps=70,
        disturbance_probability=0.65,
        disturbance_patterns=(
            "step",
            "pulse",
            "ramp",
            "bounded_random_walk",
        ),
        disturbance_max_shift=0.15,
        sensor_noise_pct=0.0,
        recovery_probability=0.0,
        state_safety_margin=0.35,
    ),
    CurriculumLevel(
        level_id="L3",
        plant_relative_std=0.07,
        operating_action_half_width=0.07,
        initial_relative_std=0.08,
        reference_patterns=(
            "single_step",
            "async_step",
            "multi_step",
            "ramp",
        ),
        reference_event_min=2,
        reference_event_max=4,
        reference_action_half_width=0.10,
        minimum_dwell_steps=60,
        disturbance_probability=0.8,
        disturbance_patterns=(
            "step",
            "pulse",
            "ramp",
            "bounded_random_walk",
            "colored_noise",
        ),
        disturbance_max_shift=0.20,
        sensor_noise_pct=0.005,
        recovery_probability=0.0,
        state_safety_margin=0.3,
    ),
    CurriculumLevel(
        level_id="L4",
        plant_relative_std=0.10,
        operating_action_half_width=0.09,
        initial_relative_std=0.10,
        reference_patterns=(
            "async_step",
            "multi_step",
            "ramp",
        ),
        reference_event_min=3,
        reference_event_max=5,
        reference_action_half_width=0.12,
        minimum_dwell_steps=50,
        disturbance_probability=0.9,
        disturbance_patterns=(
            "pulse",
            "ramp",
            "bounded_random_walk",
            "colored_noise",
            "piecewise_regime_shift",
        ),
        disturbance_max_shift=0.25,
        sensor_noise_pct=0.01,
        recovery_probability=0.35,
        state_safety_margin=0.25,
    ),
)

CURRICULUM_LEVELS = {
    level.level_id: level for level in _LEVELS
}


def get_curriculum_level(level_id: str) -> CurriculumLevel:
    """Return one canonical L0-L4 configuration."""

    try:
        return CURRICULUM_LEVELS[str(level_id)]
    except KeyError as exc:
        raise KeyError(
            f"unknown curriculum level {level_id!r}; expected one of: "
            + ", ".join(CURRICULUM_LEVELS)
        ) from exc


@dataclass(frozen=True)
class CurriculumSpec:
    """Map cumulative environment transitions to fixed levels."""

    curriculum_id: str
    boundaries: tuple[tuple[int, str], ...]
    schema_version: str = CURRICULUM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.curriculum_id, str) or not self.curriculum_id:
            raise ValueError("curriculum_id must be a non-empty string")
        if self.schema_version != CURRICULUM_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported curriculum schema {self.schema_version!r}"
            )
        if not self.boundaries or self.boundaries[0][0] != 0:
            raise ValueError("curriculum boundaries must begin at transition 0")
        previous = -1
        for transition, level_id in self.boundaries:
            if (
                isinstance(transition, bool)
                or not isinstance(transition, int)
                or transition <= previous
            ):
                raise ValueError(
                    "curriculum transition boundaries must be increasing "
                    "non-negative integers"
                )
            get_curriculum_level(level_id)
            previous = transition

    def level_for_transition(self, transition_count: int) -> CurriculumLevel:
        if (
            isinstance(transition_count, bool)
            or not isinstance(transition_count, int)
            or transition_count < 0
        ):
            raise ValueError("transition_count must be a non-negative integer")
        selected = self.boundaries[0][1]
        for boundary, level_id in self.boundaries:
            if transition_count < boundary:
                break
            selected = level_id
        return get_curriculum_level(selected)

    def metadata(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "curriculum_id": self.curriculum_id,
            "boundaries": [
                {"transition": transition, "level": level}
                for transition, level in self.boundaries
            ],
        }


QUADRUPLE_CURRICULUM_V1 = CurriculumSpec(
    curriculum_id="quadruple-regulation-curriculum-v1",
    boundaries=(
        (0, "L0"),
        (100_000, "L1"),
        (300_000, "L2"),
        (600_000, "L3"),
        (1_000_000, "L4"),
    ),
)


__all__ = [
    "CURRICULUM_LEVELS",
    "CURRICULUM_SCHEMA_VERSION",
    "QUADRUPLE_CURRICULUM_V1",
    "CurriculumLevel",
    "CurriculumSpec",
    "get_curriculum_level",
]

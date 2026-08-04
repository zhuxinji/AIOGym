"""Serializable contract between trainers and the shared lifecycle."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CHECKPOINT_SELECTIONS = frozenset(
    {"best-validation", "final", "final-fixed-dataset"}
)


@dataclass(frozen=True)
class BackendResult:
    """Small, serializable hand-off from a trainer to the shared lifecycle."""

    algorithm_id: str
    policy_path: Path
    final_step: int
    checkpoint_selection: str
    learning_curve: tuple[Mapping[str, Any], ...] = ()
    training_metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    exports: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm_id, str) or not self.algorithm_id:
            raise ValueError("algorithm_id must be a non-empty string")
        if isinstance(self.final_step, bool) or not isinstance(self.final_step, int):
            raise TypeError("final_step must be an integer")
        if self.final_step < 0:
            raise ValueError("final_step must be non-negative")
        if self.checkpoint_selection not in _CHECKPOINT_SELECTIONS:
            allowed = ", ".join(sorted(_CHECKPOINT_SELECTIONS))
            raise ValueError(f"checkpoint_selection must be one of: {allowed}")
        path = Path(self.policy_path)
        curve = tuple(dict(point) for point in self.learning_curve)
        metadata = dict(self.training_metadata)
        runtime = dict(self.runtime)
        exports = dict(self.exports)
        if any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in exports.items()
        ):
            raise TypeError("exports must map strings to strings")
        for name, value in (
            ("learning_curve", curve),
            ("training_metadata", metadata),
            ("runtime", runtime),
            ("exports", exports),
        ):
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be JSON-serializable") from exc
        object.__setattr__(self, "policy_path", path)
        object.__setattr__(self, "learning_curve", curve)
        object.__setattr__(self, "training_metadata", metadata)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "exports", exports)

    def compact_summary(self) -> dict[str, Any]:
        selected_step = (
            self.training_metadata.get("best_step")
            if self.checkpoint_selection == "best-validation"
            else self.final_step
        )
        return {
            "final_step": self.final_step,
            "checkpoint_selection": self.checkpoint_selection,
            "selected_checkpoint_step": (
                None if selected_step is None else int(selected_step)
            ),
            "runtime": dict(self.runtime),
            "exports": dict(self.exports),
        }


def validate_backend_result(plan, result: BackendResult) -> None:
    if not isinstance(result, BackendResult):
        raise TypeError("backend must return BackendResult")
    if result.algorithm_id != plan.config.algorithm_id:
        raise ValueError("backend algorithm_id does not match resolved plan")
    if result.policy_path != plan.policy_path:
        raise ValueError("backend policy_path does not match resolved plan")
    if not result.policy_path.is_file():
        raise FileNotFoundError(
            f"backend did not create selected checkpoint: {result.policy_path}"
        )


__all__ = ["BackendResult", "validate_backend_result"]

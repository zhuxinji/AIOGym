"""Complete validation-track plans and eligibility-aware selection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from aiogym.benchmarks.evaluation import evaluate_policy_on_track
from aiogym.benchmarks.tracks.registry import load_track
from aiogym.benchmarks.tracks.schema import TrackSpec
from aiogym.generation.samplers import episode_spec_from_case
from aiogym._internal.validation import seed_sequence


VALIDATION_PLAN_SCHEMA_VERSION = "aiogym.validation_plan.v1"
VALIDATION_STATE_SCHEMA_VERSION = "aiogym.validation_state.v1"


@dataclass(frozen=True)
class ValidationEpisode:
    case_id: str
    resolved_case_hash: str
    base_seed: int
    episode_spec: Any

    def metadata(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "resolved_case_hash": self.resolved_case_hash,
            "base_seed": self.base_seed,
            "episode_spec_id": self.episode_spec.episode_spec_id,
            "episode_spec_hash": self.episode_spec.resolved_hash,
        }


class ValidationEpisodePlan:
    """Fixed validation EpisodeSpecs shared by every algorithm."""

    def __init__(
        self,
        track: TrackSpec,
        *,
        base_seeds: Sequence[int],
    ) -> None:
        self.track = track
        self.base_seeds = seed_sequence(
            "validation base seeds", base_seeds
        )
        entries = []
        by_case = {}
        for case in track.resolved_cases("validation"):
            if case.case_id in by_case:
                raise ValueError(
                    "validation plan requires unique resolved case IDs"
                )
            case_entries = []
            for episode_index, seed in enumerate(self.base_seeds):
                spec = episode_spec_from_case(
                    case.profile,
                    seed=seed,
                    scenario=track.scenario,
                    goal=track.goal,
                    split="validation",
                    worker_index=0,
                    episode_index=episode_index,
                )
                entry = ValidationEpisode(
                    case_id=case.case_id,
                    resolved_case_hash=case.resolved_case_hash,
                    base_seed=seed,
                    episode_spec=spec,
                )
                entries.append(entry)
                case_entries.append(entry)
            by_case[case.case_id] = tuple(case_entries)
        self._entries = tuple(entries)
        self._by_case = by_case

    @property
    def plan_hash(self) -> str:
        canonical = json.dumps(
            self.metadata(include_entries=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def episode_specs(self, case_id: str):
        try:
            return tuple(
                entry.episode_spec for entry in self._by_case[str(case_id)]
            )
        except KeyError as exc:
            raise KeyError(
                f"validation plan has no case {case_id!r}"
            ) from exc

    def metadata(self, *, include_entries: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": VALIDATION_PLAN_SCHEMA_VERSION,
            "track_id": self.track.id,
            "track_hash": self.track.track_hash,
            "split": "validation",
            "seed_namespace": self.track.seed_namespace("validation"),
            "base_seeds": list(self.base_seeds),
        }
        if include_entries:
            data["episodes"] = [entry.metadata() for entry in self._entries]
        return data


@dataclass(frozen=True)
class CheckpointSelectionRecord:
    checkpoint_id: str
    step: int
    eligible: bool
    metric: str
    metric_direction: str
    metric_value: float
    official_score: float
    iqm_value: float
    worst_case_value: float
    intervention_cost: float
    rejection_reasons: tuple[str, ...]

    @property
    def selection_key(self) -> tuple[float, float, float]:
        sign = 1.0 if self.metric_direction == "maximize" else -1.0
        return (
            self.official_score,
            sign * self.iqm_value,
            -self.intervention_cost,
        )


class EligibilityAwareSelector:
    """Select only eligible checkpoints from validation evaluations."""

    def __init__(self) -> None:
        self.records: list[CheckpointSelectionRecord] = []
        self.best: CheckpointSelectionRecord | None = None

    def consider(
        self,
        checkpoint_id: str,
        evaluation: Mapping[str, Any],
        *,
        step: int,
    ) -> bool:
        if evaluation.get("split") != "validation":
            raise ValueError(
                "checkpoint selection accepts validation results only"
            )
        aggregate = dict(evaluation["aggregate"])
        values = np.asarray(
            aggregate.get("case_values") or (),
            dtype=np.float64,
        )
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("validation case values must be finite and non-empty")
        direction = str(aggregate["metric_direction"])
        if direction not in {"minimize", "maximize"}:
            raise ValueError("validation metric direction is invalid")
        results = tuple(evaluation.get("results") or ())
        eligible = bool(aggregate.get("ranking_eligible", True)) and all(
            bool(result.get("ranking_eligible", True))
            for result in results
        )
        reasons = tuple(
            sorted(
                {
                    str(reason)
                    for result in results
                    for reason in dict(
                        result.get("safety_gate") or {}
                    ).get("reasons", ())
                }
            )
        )
        intervention_cost = float(
            sum(
                float(
                    result.get("protection_intervention_count", 0.0)
                )
                for result in results
            )
        )
        record = CheckpointSelectionRecord(
            checkpoint_id=str(checkpoint_id),
            step=int(step),
            eligible=eligible,
            metric=str(aggregate["metric"]),
            metric_direction=direction,
            metric_value=float(aggregate["metric_value"]),
            official_score=float(
                aggregate.get("official_score", 0.0)
            ),
            iqm_value=_interquartile_mean(values),
            worst_case_value=(
                float(np.max(values))
                if direction == "minimize"
                else float(np.min(values))
            ),
            intervention_cost=intervention_cost,
            rejection_reasons=reasons,
        )
        self.records.append(record)
        if not record.eligible:
            return False
        if self.best is None or record.selection_key > self.best.selection_key:
            self.best = record
            return True
        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_STATE_SCHEMA_VERSION,
            "records": [
                {
                    **record.__dict__,
                    "rejection_reasons": list(record.rejection_reasons),
                }
                for record in self.records
            ],
            "best_checkpoint_id": (
                self.best.checkpoint_id if self.best is not None else None
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore records after validating the claimed global best."""

        if not isinstance(state, Mapping):
            raise TypeError("selector state must be a mapping")
        payload = dict(state)
        schema_version = payload.pop(
            "schema_version", VALIDATION_STATE_SCHEMA_VERSION
        )
        if schema_version != VALIDATION_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported validation state: {schema_version!r}"
            )
        raw_records = payload.pop("records", None)
        best_checkpoint_id = payload.pop("best_checkpoint_id", None)
        if payload:
            raise ValueError(
                "selector state contains unknown fields: "
                + ", ".join(sorted(payload))
            )
        if not isinstance(raw_records, (list, tuple)):
            raise TypeError("selector records must be a sequence")

        records: list[CheckpointSelectionRecord] = []
        checkpoint_ids: set[str] = set()
        for raw_record in raw_records:
            record = _selection_record_from_state(raw_record)
            if record.checkpoint_id in checkpoint_ids:
                raise ValueError(
                    "selector checkpoint IDs must be unique"
                )
            checkpoint_ids.add(record.checkpoint_id)
            records.append(record)

        expected_best = None
        for record in records:
            if not record.eligible:
                continue
            if (
                expected_best is None
                or record.selection_key > expected_best.selection_key
            ):
                expected_best = record
        expected_best_id = (
            expected_best.checkpoint_id
            if expected_best is not None
            else None
        )
        if best_checkpoint_id != expected_best_id:
            if (
                best_checkpoint_id is not None
                and best_checkpoint_id not in checkpoint_ids
            ):
                raise ValueError(
                    "best_checkpoint_id does not exist in selector records"
                )
            raise ValueError(
                "selector best checkpoint is not the best eligible record"
            )
        self.records = records
        self.best = expected_best

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "EligibilityAwareSelector":
        selector = cls()
        selector.load_state_dict(state)
        return selector


class CompleteValidationCallback:
    """Evaluate the complete fixed validation plan at checkpoint boundaries."""

    def __init__(
        self,
        track,
        *,
        base_seeds: Sequence[int],
        save_best: Callable[[str, Any, CheckpointSelectionRecord], None]
        | None = None,
        evaluate_fn=evaluate_policy_on_track,
        safety_gate_spec=None,
    ) -> None:
        self.track = (
            track if isinstance(track, TrackSpec) else load_track(track)
        )
        self.plan = ValidationEpisodePlan(
            self.track,
            base_seeds=base_seeds,
        )
        self.selector = EligibilityAwareSelector()
        self.save_best = save_best
        self.evaluate_fn = evaluate_fn
        self.safety_gate_spec = safety_gate_spec
        self.history = []

    def evaluate(self, controller, *, checkpoint_id: str, step: int):
        evaluation = evaluate_validation_policy(
            controller,
            self.plan,
            include_episodes=True,
            evaluate_fn=self.evaluate_fn,
            safety_gate_spec=self.safety_gate_spec,
        )
        improved = self.selector.consider(
            checkpoint_id,
            evaluation,
            step=step,
        )
        self.history.append(evaluation)
        if improved and self.save_best is not None:
            self.save_best(checkpoint_id, controller, self.selector.best)
        return evaluation

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_STATE_SCHEMA_VERSION,
            "track_id": self.track.id,
            "track_hash": self.track.track_hash,
            "validation_plan_hash": self.plan.plan_hash,
            "selector": self.selector.state_dict(),
            "history": list(self.history),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("validation callback state must be a mapping")
        payload = dict(state)
        schema_version = payload.pop("schema_version", None)
        if schema_version != VALIDATION_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported validation state: {schema_version!r}"
            )
        expected_identity = {
            "track_id": self.track.id,
            "track_hash": self.track.track_hash,
            "validation_plan_hash": self.plan.plan_hash,
        }
        for name, expected in expected_identity.items():
            if payload.pop(name, None) != expected:
                raise ValueError(
                    f"validation state {name} does not match current plan"
                )
        selector = EligibilityAwareSelector.from_state_dict(
            payload.pop("selector", None)
        )
        history = payload.pop("history", None)
        if not isinstance(history, (list, tuple)):
            raise TypeError("validation callback history must be a sequence")
        if payload:
            raise ValueError(
                "validation callback state contains unknown fields: "
                + ", ".join(sorted(payload))
            )
        self.selector = selector
        self.history = list(history)


def evaluate_validation_policy(
    controller,
    plan: ValidationEpisodePlan,
    *,
    include_episodes: bool = True,
    evaluate_fn=evaluate_policy_on_track,
    safety_gate_spec=None,
):
    """Run the one canonical complete-validation evaluation."""

    if not isinstance(plan, ValidationEpisodePlan):
        raise TypeError("plan must be a ValidationEpisodePlan")
    evaluation = evaluate_fn(
        controller,
        plan.track,
        base_seeds=plan.base_seeds,
        include_episodes=include_episodes,
        episode_plan=plan,
        safety_gate_spec=safety_gate_spec,
    )
    evaluation["validation_plan"] = {
        **plan.metadata(include_entries=True),
        "plan_hash": plan.plan_hash,
    }
    return evaluation


def _interquartile_mean(values) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    count = sorted_values.size
    weights = np.zeros(count, dtype=np.float64)
    for index in range(count):
        low = index / count
        high = (index + 1) / count
        weights[index] = max(0.0, min(high, 0.75) - max(low, 0.25))
    return float(np.sum(weights * sorted_values) / np.sum(weights))


def _selection_record_from_state(
    state: Mapping[str, Any],
) -> CheckpointSelectionRecord:
    if not isinstance(state, Mapping):
        raise TypeError("selector record must be a mapping")
    payload = dict(state)
    expected_fields = set(CheckpointSelectionRecord.__dataclass_fields__)
    unknown = set(payload) - expected_fields
    missing = expected_fields - set(payload)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError("invalid selector record: " + "; ".join(details))
    checkpoint_id = payload["checkpoint_id"]
    metric = payload["metric"]
    direction = payload["metric_direction"]
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("selector checkpoint_id must be non-empty")
    if not isinstance(metric, str) or not metric:
        raise ValueError("selector metric must be non-empty")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("selector metric direction is invalid")
    if type(payload["eligible"]) is not bool:
        raise TypeError("selector eligible must be boolean")
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("selector step must be a non-negative integer")
    numeric_names = (
        "metric_value",
        "official_score",
        "iqm_value",
        "worst_case_value",
        "intervention_cost",
    )
    for name in numeric_names:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"selector {name} must be numeric")
        if not np.isfinite(float(value)):
            raise ValueError(f"selector {name} must be finite")
        payload[name] = float(value)
    reasons = payload["rejection_reasons"]
    if not isinstance(reasons, (list, tuple)) or not all(
        isinstance(reason, str) for reason in reasons
    ):
        raise TypeError("selector rejection_reasons must be strings")
    payload["rejection_reasons"] = tuple(reasons)
    return CheckpointSelectionRecord(**payload)


__all__ = [
    "VALIDATION_PLAN_SCHEMA_VERSION",
    "VALIDATION_STATE_SCHEMA_VERSION",
    "CheckpointSelectionRecord",
    "CompleteValidationCallback",
    "EligibilityAwareSelector",
    "ValidationEpisode",
    "ValidationEpisodePlan",
    "evaluate_validation_policy",
]

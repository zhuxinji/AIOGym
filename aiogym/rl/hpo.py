"""Optuna integration with immutable benchmark ownership."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiogym.benchmarks.tracks.registry import load_track
from aiogym.benchmarks.tracks.schema import TrackSpec

from .config import RLTrainingConfig
from .validation import ValidationEpisodePlan


_BENCHMARK_OWNED_NAMES = frozenset(
    {
        "track",
        "track_id",
        "track_hash",
        "goal",
        "reward",
        "reward_spec",
        "reward_spec_id",
        "reward_weights",
        "safety_gate",
        "safety_thresholds",
        "case_weights",
        "training_split",
        "validation_split",
        "test_split",
        "normalization_anchors",
        "distribution_id",
    }
)
_ALGORITHM_PARAMETERS = {
    "sac": {
        "learning_rate",
        "batch_size",
        "tau",
        "utd_ratio",
        "network",
        "entropy",
        "target_entropy_scale",
        "gamma",
    },
    "td3": {
        "learning_rate",
        "batch_size",
        "tau",
        "utd_ratio",
        "network",
        "gamma",
        "policy_delay",
        "target_policy_noise",
    },
    "ppo": {
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gae_lambda",
        "clip_range",
        "entropy_coefficient",
        "value_coefficient",
        "network",
    },
    "rlpd": {
        "learning_rate",
        "utd_ratio",
        "n_critics",
        "critic_subset",
        "batch_size",
        "target_entropy_scale",
        "dataset_tier",
        "bc_steps",
        "pretrain_updates",
    },
}


@dataclass(frozen=True)
class HPOStudySpec:
    study_name: str
    storage: str
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    n_trials: int
    timeout_seconds: float | None = None

    def __post_init__(self):
        if not self.study_name or not self.storage:
            raise ValueError("study_name and storage must be non-empty")
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if not self.training_seeds or not self.validation_seeds:
            raise ValueError("HPO seed lists must be non-empty")


def apply_hpo_parameters(
    config: RLTrainingConfig,
    track: TrackSpec | str,
    parameters: Mapping[str, Any],
) -> RLTrainingConfig:
    """Apply algorithm-only suggestions without altering the benchmark."""

    resolved_track = track if isinstance(track, TrackSpec) else load_track(track)
    if config.track_id != resolved_track.id:
        raise ValueError("RL config track_id does not match HPO Track")
    requested = dict(parameters)
    forbidden = sorted(set(requested) & _BENCHMARK_OWNED_NAMES)
    if forbidden:
        raise ValueError(
            "HPO cannot override benchmark-owned field(s): "
            + ", ".join(forbidden)
        )
    allowed = _ALGORITHM_PARAMETERS.get(config.algorithm_id, set())
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(
            f"unsupported {config.algorithm_id} HPO parameter(s): "
            + ", ".join(unknown)
        )
    data = config.as_dict()
    algorithm = dict(data["algorithm"])
    algorithm.update(requested)
    data["algorithm"] = algorithm
    candidate = RLTrainingConfig.from_mapping(data)
    if candidate.track_id != resolved_track.id:
        raise AssertionError("HPO changed track identity")
    if candidate.track_id != config.track_id:
        raise AssertionError("HPO changed config track")
    return candidate


class OptunaStudyRunner:
    """Persistent validation-only Optuna study with pruning."""

    def __init__(
        self,
        *,
        track,
        base_config: RLTrainingConfig,
        study_spec: HPOStudySpec,
        suggest: Callable[[Any], Mapping[str, Any]],
        train_trial: Callable[
            [RLTrainingConfig, Any, ValidationEpisodePlan],
            Mapping[str, Any] | Iterable[Mapping[str, Any]],
        ],
    ) -> None:
        self.track = (
            track if isinstance(track, TrackSpec) else load_track(track)
        )
        if base_config.track_id != self.track.id:
            raise ValueError("base config and HPO Track do not match")
        self.base_config = base_config
        self.study_spec = study_spec
        self.suggest = suggest
        self.train_trial = train_trial
        self.validation_plan = ValidationEpisodePlan(
            self.track,
            base_seeds=study_spec.validation_seeds,
        )

    def run(self):
        try:
            import optuna
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Optuna HPO requires `pip install 'aiogym[hpo]'`"
            ) from exc
        study = optuna.create_study(
            study_name=self.study_spec.study_name,
            storage=self.study_spec.storage,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(),
        )

        def objective(trial):
            parameters = dict(self.suggest(trial))
            candidate = apply_hpo_parameters(
                self.base_config,
                self.track,
                parameters,
            )
            trial.set_user_attr("config_hash", candidate.config_hash)
            trial.set_user_attr("track_id", self.track.id)
            trial.set_user_attr("track_hash", self.track.track_hash)
            trial.set_user_attr("reward_spec_id", self.track.reward_spec_id)
            trial.set_user_attr(
                "training_seeds",
                list(self.study_spec.training_seeds),
            )
            trial.set_user_attr(
                "validation_plan_hash",
                self.validation_plan.plan_hash,
            )
            outcomes = self.train_trial(
                candidate,
                trial,
                self.validation_plan,
            )
            evaluations = (
                [outcomes] if isinstance(outcomes, Mapping) else outcomes
            )
            best_utility = float("-inf")
            for step, evaluation in enumerate(evaluations):
                utility = _validation_utility(evaluation)
                best_utility = max(best_utility, utility)
                trial.report(utility, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if best_utility == float("-inf"):
                raise ValueError("HPO trial produced no validation evaluations")
            return best_utility

        study.optimize(
            objective,
            n_trials=self.study_spec.n_trials,
            timeout=self.study_spec.timeout_seconds,
        )
        return study


def _validation_utility(evaluation: Mapping[str, Any]) -> float:
    if evaluation.get("split") != "validation":
        raise ValueError("HPO objective may read validation results only")
    aggregate = dict(evaluation["aggregate"])
    results = tuple(evaluation.get("results") or ())
    eligible = bool(aggregate.get("ranking_eligible", True)) and all(
        bool(result.get("ranking_eligible", True))
        for result in results
    )
    if not eligible:
        return float("-inf")
    if "official_score" in aggregate:
        value = float(aggregate["official_score"])
        if not np.isfinite(value):
            raise ValueError("HPO official score must be finite")
        return value
    value = float(aggregate["metric_value"])
    if not np.isfinite(value):
        raise ValueError("HPO validation metric must be finite")
    direction = aggregate["metric_direction"]
    if direction == "maximize":
        return value
    if direction == "minimize":
        return -value
    raise ValueError("HPO validation metric direction is invalid")


__all__ = [
    "HPOStudySpec",
    "OptunaStudyRunner",
    "apply_hpo_parameters",
]

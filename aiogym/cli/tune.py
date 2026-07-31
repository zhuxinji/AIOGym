"""Validation-only Optuna tuning through the unified runner."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.hpo import HPOStudySpec, OptunaStudyRunner
from aiogym.rl.runner import run_experiment


TUNE_CONFIG_SCHEMA_VERSION = "aiogym.tune.v1"


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Tune SAC, TD3, or PPO on one fixed validation plan.",
    )
    parser.add_argument("--config", required=True)
    return parser


def main(argv=None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    declaration = _load_declaration(args.config)
    base_config = RLTrainingConfig.from_mapping(declaration["training"])
    if base_config.algorithm_id not in {"ppo", "sac", "td3"}:
        raise ValueError("stable tune supports only SAC, TD3, and PPO")
    study_data = dict(declaration["study"])
    study_spec = HPOStudySpec(
        study_name=str(study_data["name"]),
        storage=str(study_data["storage"]),
        training_seeds=tuple(study_data["training_seeds"]),
        validation_seeds=tuple(study_data["validation_seeds"]),
        n_trials=int(study_data["n_trials"]),
        timeout_seconds=study_data.get("timeout_seconds"),
    )
    search_space = dict(declaration["search_space"])

    def suggest(trial):
        return {
            name: _suggest_parameter(trial, name, dict(spec))
            for name, spec in sorted(search_space.items())
        }

    def train_trial(candidate, trial, validation_plan):
        evaluations = []
        base_output = dict(candidate.output)
        base_name = str(
            base_output.get("name")
            or f"{candidate.algorithm_id}-{candidate.track_id}"
        )
        for seed in study_spec.training_seeds:
            output = {
                **base_output,
                "name": (
                    f"{base_name}-trial{trial.number}-seed{seed}"
                ),
            }
            result = run_experiment(
                replace(
                    candidate,
                    training_seed=int(seed),
                    validation_seeds=study_spec.validation_seeds,
                    output=output,
                ),
                validation_plan=validation_plan,
            )
            if not result.validation:
                raise ValueError(
                    "training lifecycle did not produce validation artifacts"
                )
            evaluations.append(result.validation)
        return evaluations

    runner = OptunaStudyRunner(
        track=base_config.track_id,
        base_config=base_config,
        study_spec=study_spec,
        suggest=suggest,
        train_trial=train_trial,
    )
    study = runner.run()
    result = {
        "schema_version": "aiogym.tune_result.v1",
        "study_name": study.study_name,
        "track_id": base_config.track_id,
        "algorithm_id": base_config.algorithm_id,
        "best_value": study.best_value,
        "best_params": dict(study.best_params),
        "trial_count": len(study.trials),
        "validation_plan_hash": runner.validation_plan.plan_hash,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _load_declaration(path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise TypeError("tune config must be a mapping")
    required = {"schema_version", "training", "study", "search_space"}
    if set(value) != required:
        raise ValueError(
            "tune config requires exactly: "
            + ", ".join(sorted(required))
        )
    if value["schema_version"] != TUNE_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported tune config schema")
    for name in ("training", "study", "search_space"):
        if not isinstance(value[name], Mapping):
            raise TypeError(f"tune config {name} must be a mapping")
    return dict(value)


def _suggest_parameter(trial, name: str, spec: dict):
    kind = spec.pop("type", None)
    if kind == "float":
        return trial.suggest_float(name, **spec)
    if kind == "int":
        return trial.suggest_int(name, **spec)
    if kind == "categorical":
        choices = spec.pop("choices", None)
        if spec or not isinstance(choices, list) or not choices:
            raise ValueError(
                f"categorical search parameter {name!r} requires choices"
            )
        return trial.suggest_categorical(name, choices)
    raise ValueError(
        f"search parameter {name!r} has unsupported type {kind!r}"
    )


__all__ = ["TUNE_CONFIG_SCHEMA_VERSION", "build_parser", "main"]

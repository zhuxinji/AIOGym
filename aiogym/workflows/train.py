"""Backend-neutral train, save, and load workflow."""
from __future__ import annotations

import copy
import json
import logging
import math
import shutil
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aiogym.core.io import jsonable, write_json
from aiogym.rl.algorithms import (
    TrainingStep,
    get_algorithm,
    resolve_algorithm_kwargs,
)
from aiogym.rl.behavior_cloning import load_demonstrations
from aiogym.rl.datasets import load_training_dataset, training_dataset_metadata

from ._metadata import environment_metadata, validate_environment_compatibility
from .compare import render_trajectory_svg
from ._checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    backend_runtime_metadata,
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from .evaluate import (
    DEFAULT_VALIDATION_SETTLING_FRACTION,
    _evaluate,
    _validation_summary,
    evaluate,
)
from .training_curve import (
    TRAINING_CURVE_SCHEMA_VERSION,
    _load_curve,
    _validate_curve,
    _validation_rows,
    plot_training_curve,
)


TRAINING_SCHEMA_VERSION = "aiogym.training.v14"
TRAINING_EVALUATION_SCHEMA_VERSION = "aiogym.training_evaluation.v4"
DEFAULT_VALIDATION_SEEDS = tuple(range(1_000, 1_020))
_LOGGER = logging.getLogger(__name__)


def train(
    *,
    env,
    algorithm: str,
    steps: int = 500_000,
    output: str | Path,
    seed: int | None = None,
    algorithm_kwargs: Mapping | None = None,
    record_every: int = 500,
    evaluation_env=None,
    evaluate_every: int | None = 5_000,
    dataset: str | Path | None = None,
    behavior_cloning_epochs: int | None = None,
    behavior_cloning_batch_size: int = 256,
    behavior_cloning_learning_rate: float = 3e-4,
    resume_from: str | Path | None = None,
):
    """Train or continue for 500k steps, validating every 5k by default.

    Supply a separate ``evaluation_env`` or set ``evaluate_every=None`` to
    disable validation. The caller retains ownership of both environments.
    The output must be new or empty apart from CLI logs, status, and frozen
    JSON inputs; existing training artifacts are never overwritten.
    """

    if env.unwrapped.benchmark is not None:
        raise ValueError("train does not accept a benchmark environment")
    backend = get_algorithm(algorithm)
    key = backend.id
    resolved_steps = _positive_integer("steps", steps)
    resolved_record_every = _positive_integer("record_every", record_every)
    resolved_evaluate_every = (
        None
        if evaluate_every is None
        else _positive_integer("evaluate_every", evaluate_every)
    )
    _validate_evaluation_env(env, evaluation_env, resolved_evaluate_every)
    requested_kwargs = copy.deepcopy(
        {} if algorithm_kwargs is None else dict(algorithm_kwargs)
    )
    if any(not isinstance(name, str) for name in requested_kwargs):
        raise TypeError("algorithm_kwargs keys must be strings")
    serialized_kwargs = jsonable(requested_kwargs)
    if not isinstance(serialized_kwargs, dict):
        raise TypeError("algorithm_kwargs must be a mapping")

    resume_path = None if resume_from is None else Path(resume_from)
    initial_steps = 0
    resumed_model = None
    checkpoint_training = None
    if resume_path is not None:
        if behavior_cloning_epochs is not None:
            raise ValueError("continued training does not accept behavior cloning")
        if dataset is not None and not backend.requires_dataset:
            raise ValueError(
                f"continued {key} training does not accept a Dataset"
            )
        (
            resume_path,
            loaded_backend,
            resumed_model,
            checkpoint_training,
        ) = load_training_checkpoint(resume_path, env=env, algorithm=key)
        backend = loaded_backend
        initial_steps = checkpoint_training["completed_steps"]
        checkpoint_seed = checkpoint_training["seed"]
        if seed is not None:
            requested_seed = _nonnegative_integer("seed", seed)
            if requested_seed != checkpoint_seed:
                raise ValueError(
                    "continued training seed must match checkpoint seed "
                    f"{checkpoint_seed}"
                )
        resolved_seed = checkpoint_seed
        resolved_kwargs = dict(checkpoint_training["algorithm_kwargs"])
        if algorithm_kwargs is not None:
            requested_resolved_kwargs = resolve_algorithm_kwargs(
                backend,
                steps=resolved_steps,
                values=serialized_kwargs,
            )
            if requested_resolved_kwargs != resolved_kwargs:
                raise ValueError(
                    "continued training algorithm_kwargs must match the checkpoint"
                )
    else:
        resolved_seed = _nonnegative_integer("seed", 0 if seed is None else seed)
        resolved_kwargs = resolve_algorithm_kwargs(
            backend,
            steps=resolved_steps,
            values=serialized_kwargs,
        )

    dataset_path, cloning = _dataset_inputs(
        backend=backend,
        dataset=dataset,
        epochs=behavior_cloning_epochs,
        batch_size=behavior_cloning_batch_size,
        learning_rate=behavior_cloning_learning_rate,
    )
    training_dataset = (
        None
        if dataset_path is None
        else load_training_dataset(dataset_path, env=env)
    )
    dataset_metadata = (
        None
        if training_dataset is None
        else training_dataset_metadata(training_dataset)
    )
    if resume_path is not None and backend.requires_dataset:
        expected_dataset = checkpoint_training["dataset"]
        if expected_dataset is None or "content_sha256" not in expected_dataset:
            raise ValueError(
                "checkpoint has no Dataset content digest; original data identity "
                "cannot be verified for continuation"
            )
        if dataset_metadata["content_sha256"] != expected_dataset["content_sha256"]:
            raise ValueError("continued training Dataset content must match the checkpoint Dataset")
    history = (
        None if resume_path is None else _resume_history(
            resume_path, checkpoint_training, evaluation_env=evaluation_env,
        )
    )
    demonstration_data = (
        None
        if cloning is None
        else load_demonstrations(training_dataset)
    )

    directory = Path(output)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"training output is not a directory: {directory}")
    cli_files = {"train.log", "status.json", "parameters.json", "algorithm-kwargs.json"}
    if directory.exists() and any(
        path.name not in cli_files or not path.is_file() or path.is_symlink()
        for path in directory.iterdir()
    ):
        raise FileExistsError(
            f"refusing to create training output in non-empty directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Training initialized", extra={
        "phase": "setup", "initial_steps": initial_steps, "seed": resolved_seed,
    })

    model = (
        resumed_model
        if resumed_model is not None
        else backend.create(
            env=env,
            seed=resolved_seed,
            algorithm_kwargs=copy.deepcopy(resolved_kwargs),
        )
    )
    behavior_cloning_report = None
    if cloning is not None:
        _LOGGER.info("Cloning demonstrations", extra={"phase": "behavior-cloning"})
        observations, actions, source = demonstration_data
        behavior_cloning_report = backend.behavior_cloning(
            model,
            observations,
            actions,
            epochs=cloning["epochs"],
            batch_size=cloning["batch_size"],
            learning_rate=cloning["learning_rate"],
            seed=resolved_seed,
            source=source,
        )
        if not isinstance(behavior_cloning_report, Mapping):
            raise TypeError("algorithm backend behavior_clone must return a mapping")
    curve_recorder = _TrainingCurveRecorder(
        resolved_record_every,
        initial_steps=initial_steps,
        history=None if history is None else history["curve"],
    )
    checkpoint_dataset = dataset_metadata if backend.requires_dataset else None

    def checkpoint_state(completed_steps):
        return {
            "completed_steps": int(completed_steps),
            "seed": resolved_seed,
            "algorithm_kwargs": resolved_kwargs,
            "dataset": checkpoint_dataset,
        }

    evaluation_recorder = None

    def checkpoint_history(*, is_best=False):
        return {
            "curve": curve_recorder.snapshot(),
            "evaluation": (
                None if evaluation_recorder is None else evaluation_recorder.payload()
            ),
            "best_checkpoint": (
                None if is_best or evaluation_recorder is None
                else str((directory / "best/model.zip").resolve())
            ),
        }

    if resolved_evaluate_every is not None:
        best_checkpoint = directory / "best/model.zip"
        if history is not None and history["evaluation"] is not None:
            best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(history["best_checkpoint"], best_checkpoint)
        evaluation_recorder = _TrainingEvaluationRecorder(
            backend=backend,
            model=model,
            env=evaluation_env,
            checkpoint_env=env,
            evaluate_every=resolved_evaluate_every,
            seeds=DEFAULT_VALIDATION_SEEDS,
            best_checkpoint=best_checkpoint,
            initial_steps=initial_steps,
            checkpoint_state=checkpoint_state,
            checkpoint_history=checkpoint_history,
            history=None if history is None else history["evaluation"],
        )
        evaluation_recorder.start()
    if curve_recorder.snapshot() is not None:
        write_json(directory / "training_curve.json", curve_recorder.snapshot())

    def record_step(step: TrainingStep) -> None:
        cumulative = TrainingStep(
            step=initial_steps + step.step,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
        )
        flushed = curve_recorder.on_step(cumulative)
        if flushed or (
            resolved_evaluate_every is not None
            and cumulative.step % resolved_evaluate_every == 0
        ):
            write_json(
                directory / "training_curve.json", curve_recorder.snapshot(),
                overwrite=True,
            )
        if evaluation_recorder is not None:
            evaluation_recorder.on_step(cumulative)

    _LOGGER.info("Learning from environment", extra={"phase": "training"})
    actual_steps = backend.learn(
        model,
        steps=resolved_steps,
        dataset=training_dataset,
        on_step=record_step,
    )
    added_steps = _positive_integer("backend actual_steps", actual_steps)
    total_steps = initial_steps + added_steps
    curve_recorder.finish(total_steps)
    if evaluation_recorder is not None:
        evaluation_recorder.finish(total_steps)
    _LOGGER.info("Saving final checkpoint", extra={"phase": "saving"})
    curve = curve_recorder.payload()
    checkpoint = directory / "model.zip"
    save_checkpoint(
        backend,
        model,
        checkpoint,
        env=env,
        training=checkpoint_state(total_steps),
        history=checkpoint_history(),
    )

    git_executable = shutil.which("git")
    git_commit = None
    if git_executable is not None:
        completed = subprocess.run(
            [git_executable, "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode == 0:
            git_commit = completed.stdout.strip()

    evaluation_metadata = None
    evaluation_history = None
    evaluation_history_path = None
    best_checkpoint = None
    best_tracking_figure = None
    if evaluation_recorder is not None:
        evaluation_history = evaluation_recorder.payload()
        evaluation_history_path = write_json(
            directory / "evaluation_history.json",
            evaluation_history,
            overwrite=True,
        )
        best_checkpoint = directory / "best" / "model.zip"
        if not best_checkpoint.is_file():
            raise FileNotFoundError(
                f"training evaluation did not create checkpoint: {best_checkpoint}"
            )
        best_policy = load_policy(best_checkpoint, env=evaluation_env)
        _LOGGER.info("Evaluating best checkpoint", extra={"phase": "validation"})
        best_evaluation = evaluate(
            env=evaluation_env,
            policy=best_policy,
            seeds=DEFAULT_VALIDATION_SEEDS,
        )
        _LOGGER.info("Writing training artifacts", extra={"phase": "saving"})
        best_tracking_figure = best_checkpoint.parent / "tracking.svg"
        trajectory_report = {
            "environment": best_evaluation["environment"],
            "trajectory_schema": best_evaluation["trajectory_schema"],
            "seeds": best_evaluation["seeds"],
            "trajectory_seed": best_evaluation["seeds"][0],
            "evaluations": {f"{key} best": best_evaluation},
        }
        best_tracking_figure.write_text(
            render_trajectory_svg(
                trajectory_report,
                title=(
                    f"{best_evaluation['environment']['scenario']} {key.upper()} "
                    "best policy - validation cases"
                ),
            ),
            encoding="utf-8",
        )
        evaluation_metadata = {
            "environment": environment_metadata(evaluation_env),
            "evaluate_every": resolved_evaluate_every,
            "seeds": list(DEFAULT_VALIDATION_SEEDS),
            "ranking_metric": evaluation_history["ranking_metric"],
            "selection_order": evaluation_history["selection_order"],
            "settling_fraction": evaluation_history["settling_fraction"],
            "best_step": evaluation_history["best_step"],
            "best_mean_return": evaluation_history["best_mean_return"],
            "best_median_return": evaluation_history["best_median_return"],
            "best_p10_return": evaluation_history["best_p10_return"],
            "best_safe_completion": evaluation_history[
                "best_safe_completion"
            ],
            "best_control_success": evaluation_history[
                "best_control_success"
            ],
            "tracking_figure": "best/tracking.svg",
        }

    metadata = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": key,
        "steps": resolved_steps,
        "initial_steps": initial_steps,
        "added_steps": added_steps,
        "actual_steps": curve["actual_steps"],
        "resume_from": (
            None if resume_path is None else str(resume_path.resolve())
        ),
        "seed": resolved_seed,
        "record_every": resolved_record_every,
        "algorithm_kwargs": resolved_kwargs,
        "environment": environment_metadata(env),
        "dataset": dataset_metadata,
        "behavior_cloning": (
            None
            if behavior_cloning_report is None
            else {
                "dataset": behavior_cloning_report["dataset"],
                "transition_count": behavior_cloning_report["transition_count"],
                "action_field": behavior_cloning_report["action_field"],
                "epochs": behavior_cloning_report["epochs"],
                "batch_size": behavior_cloning_report["batch_size"],
                "learning_rate": behavior_cloning_report["learning_rate"],
                "artifact": "behavior_cloning.json",
            }
        ),
        "evaluation": evaluation_metadata,
        "runtime": backend_runtime_metadata(backend),
        "git_commit": git_commit,
    }
    write_json(directory / "metadata.json", metadata)
    behavior_cloning_path = (
        None
        if behavior_cloning_report is None
        else write_json(
            directory / "behavior_cloning.json", behavior_cloning_report
        )
    )
    curve_path = write_json(directory / "training_curve.json", curve, overwrite=True)
    curve_figure = plot_training_curve(
        curve,
        output=directory / "training_curve.svg",
        evaluation_history=evaluation_history,
        control_dt=float(env.unwrapped.control_dt),
    )
    return {
        **metadata,
        "path": str(directory.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "best_checkpoint": (
            None if best_checkpoint is None else str(best_checkpoint.resolve())
        ),
        "best_tracking_figure": (
            None
            if best_tracking_figure is None
            else str(best_tracking_figure.resolve())
        ),
        "evaluation_history": (
            None
            if evaluation_history_path is None
            else str(evaluation_history_path.resolve())
        ),
        "behavior_cloning_artifact": (
            None
            if behavior_cloning_path is None
            else str(behavior_cloning_path.resolve())
        ),
        "training_curve": str(curve_path.resolve()),
        "training_curve_figure": str(curve_figure.resolve()),
    }


def _resume_history(path, training, *, evaluation_env):
    """Restore the history belonging to this checkpoint, including older runs."""
    step = training["completed_steps"]
    if "history" in training:
        history = copy.deepcopy(training["history"])
    else:
        # Older containers keep records next to the final checkpoint.
        directory = path.parent.parent if path.parent.name == "best" else path.parent
        curve = None
        if step:
            curve_path = directory / "training_curve.json"
            if not curve_path.is_file():
                raise ValueError(
                    "checkpoint has no saved training history; keep the original "
                    f"training_curve.json at {curve_path} to continue without losing records"
                )
            curve = _load_curve(curve_path)
            _validate_curve(curve)
            curve["records"] = [r for r in curve["records"] if r["end_step"] <= step]
            curve["episodes"] = [r for r in curve["episodes"] if r["end_step"] <= step]
            curve["actual_steps"] = step
        evaluation_path = directory / "evaluation_history.json"
        evaluation = _load_curve(evaluation_path) if evaluation_path.is_file() else None
        if evaluation is not None:
            evaluation["records"] = [r for r in evaluation["records"] if r["step"] <= step]
            if not any(r["step"] == evaluation["best_step"] for r in evaluation["records"]):
                evaluation["best_step"] = step
        history = {
            "curve": curve,
            "evaluation": evaluation,
            "best_checkpoint": (
                None if path.parent.name == "best"
                else str(directory / "best/model.zip")
            ),
        }
    if not isinstance(history, dict) or set(history) != {"curve", "evaluation", "best_checkpoint"}:
        raise ValueError("checkpoint training history requires curve, evaluation and best_checkpoint")
    curve, evaluation = history["curve"], history["evaluation"]
    if curve is None:
        if step:
            raise ValueError("checkpoint is missing its training curve history")
    else:
        _validate_curve(curve)
        if curve["actual_steps"] != step:
            raise ValueError("training history must end at the checkpoint step")
    if (evaluation is None) != (evaluation_env is None):
        if step == 0 and evaluation is None:
            return history
        raise ValueError("continued training must preserve whether validation is enabled")
    if evaluation is not None:
        if evaluation.get("schema_version") not in {
            "aiogym.training_evaluation.v2", "aiogym.training_evaluation.v3",
            TRAINING_EVALUATION_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported checkpoint validation history schema")
        _validation_rows(
            evaluation, curve or {"initial_steps": 0, "actual_steps": 0},
            control_dt=float(evaluation_env.unwrapped.control_dt),
            settling_fraction=DEFAULT_VALIDATION_SETTLING_FRACTION,
        )
        source = history["best_checkpoint"]
        if source is None:
            source = path
        else:
            local_best = path.parent / "best/model.zip"
            source = local_best if local_best.is_file() else Path(source)
        if not source.is_file():
            raise ValueError(f"saved best checkpoint is required to preserve selection: {source}")
        with zipfile.ZipFile(source) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        if manifest["policy"]["training"]["completed_steps"] != evaluation["best_step"]:
            raise ValueError("saved best checkpoint does not match the validation history")
        if evaluation.get("settling_fraction") != DEFAULT_VALIDATION_SETTLING_FRACTION:
            # Old validation points remain diagnostics. Only retained weights can
            # compete under a new rule; a metric history cannot restore a model.
            available = {evaluation["best_step"]: source, step: path}
            best = None
            for record in evaluation["records"]:
                if (
                    evaluation.get("schema_version") != TRAINING_EVALUATION_SCHEMA_VERSION
                    and getattr(evaluation_env.unwrapped.model, "time_unit", "s") == "h"
                ):
                    for episode in record["episodes"]:
                        episode["metrics"]["energy"] *= 3600
                record.update(_validation_summary(
                    record["episodes"], control_dt=float(evaluation_env.unwrapped.control_dt),
                ))
                record["selection_eligible"] = record["step"] in available
                if _is_better_training_evaluation(record, best):
                    best = record
            source = available[best["step"]]
            evaluation["best_step"] = best["step"]
            for name in ("mean_return", "median_return", "p10_return", "safe_completion", "control_success"):
                evaluation[f"best_{name}"] = best[name]
            evaluation["schema_version"] = TRAINING_EVALUATION_SCHEMA_VERSION
            evaluation.pop("settling_window", None)
            evaluation["settling_fraction"] = DEFAULT_VALIDATION_SETTLING_FRACTION
        history["best_checkpoint"] = source
    return history


def _dataset_inputs(*, backend, dataset, epochs, batch_size, learning_rate):
    if dataset is None:
        if backend.requires_dataset:
            raise ValueError(f"algorithm backend {backend.id} requires dataset")
        if epochs is not None:
            raise ValueError("behavior_cloning_epochs requires dataset")
        return None, None
    if backend.behavior_cloning is None and not backend.requires_dataset:
        raise ValueError(
            f"algorithm backend {backend.id} does not consume a training Dataset"
        )
    if epochs is None and backend.behavior_cloning is not None:
        raise ValueError("dataset requires behavior_cloning_epochs for this algorithm")
    if epochs is None:
        return Path(dataset), None
    if backend.behavior_cloning is None:
        raise ValueError(
            f"algorithm backend {backend.id} does not support behavior cloning"
        )
    return Path(dataset), {
        "epochs": _positive_integer("behavior_cloning_epochs", epochs),
        "batch_size": _positive_integer(
            "behavior_cloning_batch_size", batch_size
        ),
        "learning_rate": _positive_float(
            "behavior_cloning_learning_rate", learning_rate
        ),
    }


def load_policy(
    checkpoint: str | Path,
    *,
    env,
):
    """Load one self-describing AIO-Gym ``model.zip`` policy."""

    return load_checkpoint(checkpoint, env=env)


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _validate_evaluation_env(training_env, evaluation_env, evaluate_every):
    if evaluate_every is None:
        if evaluation_env is not None:
            raise ValueError("evaluation_env requires evaluate_every")
        return
    if evaluation_env is None:
        raise ValueError(
            "evaluate_every requires evaluation_env; provide a separate randomized "
            "environment or set evaluate_every=None to disable validation"
        )
    if evaluation_env is training_env:
        raise ValueError("evaluation_env must be separate from the training env")
    if evaluation_env.unwrapped.benchmark is not None:
        raise ValueError("training evaluation does not accept a Benchmark env")
    training_metadata = environment_metadata(training_env)
    evaluation_metadata = environment_metadata(evaluation_env)
    if not evaluation_metadata["randomize"]:
        raise ValueError("training evaluation env must use randomize=True")
    for channel in ("disturbance", "noise", "delay", "fault"):
        if evaluation_metadata[channel] is not None:
            raise ValueError(
                f"training evaluation env must not enable {channel}"
            )
    validate_environment_compatibility(training_metadata, evaluation_env)
    if training_env.observation_space != evaluation_env.observation_space:
        raise ValueError("evaluation observation space must match training")
    if training_env.action_space != evaluation_env.action_space:
        raise ValueError("evaluation action space must match training")


class _TrainingEvaluationRecorder:
    def __init__(
        self,
        *,
        backend,
        model,
        env,
        checkpoint_env,
        evaluate_every,
        seeds,
        best_checkpoint,
        initial_steps,
        checkpoint_state,
        checkpoint_history,
        history=None,
    ):
        self._backend = backend
        self._model = model
        self._env = env
        self._checkpoint_env = checkpoint_env
        self._evaluate_every = evaluate_every
        self._seeds = tuple(seeds)
        self._best_checkpoint = best_checkpoint
        self._initial_steps = initial_steps
        self._checkpoint_state = checkpoint_state
        self._checkpoint_history = checkpoint_history
        self._records = [] if history is None else copy.deepcopy(history["records"])
        self._best_index = None
        for index, record in enumerate(self._records):
            if _is_better_training_evaluation(
                record, None if self._best_index is None else self._records[self._best_index],
            ):
                self._best_index = index

    def start(self):
        if not self._records:
            self._evaluate(self._initial_steps)
        else:
            self._persist()

    def on_step(self, event: TrainingStep):
        if event.step % self._evaluate_every == 0:
            self._evaluate(event.step)

    def finish(self, actual_steps):
        if not self._records or self._records[-1]["step"] != actual_steps:
            self._evaluate(actual_steps)

    def _evaluate(self, step):
        _LOGGER.info("Validating checkpoint", extra={"phase": "validation"})
        policy = self._backend.policy(
            self._model,
            checkpoint=self._best_checkpoint,
        )
        result = _evaluate(env=self._env, policy=policy, seeds=self._seeds,
                           include_trajectories=False)
        episodes = result["episodes"]
        summary = _validation_summary(
            episodes, control_dt=float(self._env.unwrapped.control_dt),
        )
        record = {
            "step": int(step),
            **summary,
            "episodes": [
                {
                    "seed": int(episode["seed"]),
                    "return": float(episode["return"]),
                    "length": int(episode["length"]),
                    "terminated": bool(episode["terminated"]),
                    "truncated": bool(episode["truncated"]),
                    "metrics": dict(episode["metrics"]),
                    "episode_spec": copy.deepcopy(episode["episode_spec"]),
                    "episode_family": episode["episode_family"],
                    "episode_parameters": copy.deepcopy(
                        episode["episode_parameters"]
                    ),
                    "runtime_variation": copy.deepcopy(
                        episode["runtime_variation"]
                    ),
                }
                for episode in episodes
            ],
        }
        self._records.append(record)
        improved = _is_better_training_evaluation(
            record,
            None if self._best_index is None else self._records[self._best_index],
        )
        if improved:
            self._best_index = len(self._records) - 1
        self._persist()
        if improved:
            self._best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                self._backend,
                self._model,
                self._best_checkpoint,
                env=self._checkpoint_env,
                training=self._checkpoint_state(step),
                history=self._checkpoint_history(is_best=True),
                overwrite=True,
            )

        _LOGGER.info("Validation complete", extra={"phase": "training"})

    def _persist(self):
        write_json(
            self._best_checkpoint.parent.parent / "evaluation_history.json",
            self.payload(), overwrite=True,
        )

    def payload(self):
        if not self._records or self._best_index is None:
            raise RuntimeError("training evaluation produced no records")
        best = self._records[self._best_index]
        return {
            "schema_version": TRAINING_EVALUATION_SCHEMA_VERSION,
            "evaluate_every": int(self._evaluate_every),
            "seeds": list(self._seeds),
            "settling_fraction": DEFAULT_VALIDATION_SETTLING_FRACTION,
            "ranking_metric": {
                "name": "return",
                "aggregate": "mean",
                "direction": "maximize",
            },
            "selection_order": [
                {"name": "safe_completion", "direction": "maximize"},
                {"name": "control_success", "direction": "maximize"},
                {"name": "mean_return", "direction": "maximize"},
            ],
            "best_step": best["step"],
            "best_mean_return": best["mean_return"],
            "best_median_return": best["median_return"],
            "best_p10_return": best["p10_return"],
            "best_safe_completion": best["safe_completion"],
            "best_control_success": best["control_success"],
            "records": list(self._records),
        }


def _is_better_training_evaluation(candidate, best):
    if not candidate.get("selection_eligible", True):
        return False
    if best is None:
        return True
    if candidate["safe_completion"] != best["safe_completion"]:
        return candidate["safe_completion"] > best["safe_completion"]
    if candidate["control_success"] != best["control_success"]:
        return candidate["control_success"] > best["control_success"]
    return candidate["mean_return"] > best["mean_return"]


class _TrainingCurveRecorder:
    def __init__(self, record_every, *, initial_steps=0, history=None):
        self._record_every = record_every
        self._initial_steps = initial_steps if history is None else history["initial_steps"]
        self._window_start = initial_steps
        self._window_rewards = []
        self._window_completed = 0
        self._window_terminated = 0
        self._window_truncated = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._records = [] if history is None else copy.deepcopy(history["records"])
        self._episodes = [] if history is None else copy.deepcopy(history["episodes"])
        self._last_step = initial_steps
        self._actual_steps = None

    def on_step(self, event: TrainingStep):
        if not isinstance(event, TrainingStep):
            raise TypeError("algorithm backend on_step requires TrainingStep")
        if (
            isinstance(event.step, bool)
            or not isinstance(event.step, int)
            or event.step != self._last_step + 1
        ):
            raise ValueError(
                "algorithm backend must report consecutive TrainingStep.step values"
            )
        reward = float(event.reward)
        if not math.isfinite(reward):
            raise FloatingPointError("training reward must be finite")
        if not isinstance(event.terminated, bool) or not isinstance(
            event.truncated, bool
        ):
            raise TypeError("TrainingStep terminated and truncated must be bool")
        self._last_step = event.step
        self._window_rewards.append(reward)
        self._episode_return += reward
        self._episode_length += 1
        if event.terminated or event.truncated:
            outcome = "truncated" if event.truncated else "terminated"
            self._episodes.append(
                {
                    "end_step": int(event.step),
                    "return": float(self._episode_return),
                    "length": int(self._episode_length),
                    "outcome": outcome,
                }
            )
            self._window_completed += 1
            self._window_truncated += int(event.truncated)
            self._window_terminated += int(event.terminated and not event.truncated)
            self._episode_return = 0.0
            self._episode_length = 0
        if event.step - self._window_start == self._record_every:
            self._flush_window(event.step)
            return True
        return False

    def finish(self, actual_steps):
        if actual_steps != self._last_step:
            raise ValueError(
                "algorithm backend learn() actual_steps must equal its last "
                "TrainingStep.step"
            )
        if self._window_rewards:
            self._flush_window(actual_steps)
        self._actual_steps = actual_steps

    def _flush_window(self, end_step):
        rewards = np.asarray(self._window_rewards, dtype=float)
        self._records.append(
            {
                "start_step": int(self._window_start),
                "end_step": int(end_step),
                "transition_count": int(rewards.size),
                "mean_reward": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "minimum_reward": float(np.min(rewards)),
                "maximum_reward": float(np.max(rewards)),
                "completed_episodes": int(self._window_completed),
                "terminated_episodes": int(self._window_terminated),
                "truncated_episodes": int(self._window_truncated),
            }
        )
        self._window_start = int(end_step)
        self._window_rewards = []
        self._window_completed = 0
        self._window_terminated = 0
        self._window_truncated = 0

    def payload(self):
        if self._actual_steps is None:
            raise RuntimeError("training curve was requested before training ended")
        if not self._records:
            raise RuntimeError("training produced no curve records")
        return {
            "schema_version": TRAINING_CURVE_SCHEMA_VERSION,
            "record_every": int(self._record_every),
            "initial_steps": int(self._initial_steps),
            "actual_steps": int(self._actual_steps),
            "records": list(self._records),
            "episodes": list(self._episodes),
        }

    def snapshot(self):
        if self._last_step == self._initial_steps:
            return None
        snapshot = copy.copy(self)
        snapshot._records = list(self._records)
        if self._window_rewards:
            snapshot._flush_window(self._last_step)
        snapshot._actual_steps = self._last_step
        return snapshot.payload()


__all__ = ["TRAINING_SCHEMA_VERSION", "load_policy", "train"]

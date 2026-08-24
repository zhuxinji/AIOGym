"""Backend-neutral train, save, and load workflow."""
from __future__ import annotations

import copy
import math
import shutil
import subprocess
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
from .evaluate import evaluate
from .training_curve import (
    TRAINING_CURVE_SCHEMA_VERSION,
    plot_training_curve,
)


TRAINING_SCHEMA_VERSION = "aiogym.training.v10"
TRAINING_EVALUATION_SCHEMA_VERSION = "aiogym.training_evaluation.v1"


def train(
    *,
    env,
    algorithm: str,
    steps: int,
    output: str | Path,
    seed: int | None = None,
    algorithm_kwargs: Mapping | None = None,
    record_every: int = 500,
    evaluation_env=None,
    evaluate_every: int | None = None,
    evaluation_seed: int = 0,
    dataset: str | Path | None = None,
    behavior_cloning_epochs: int | None = None,
    behavior_cloning_batch_size: int = 256,
    behavior_cloning_learning_rate: float = 3e-4,
    resume_from: str | Path | None = None,
):
    """Train or continue one algorithm without taking ownership of ``env``."""

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
    resolved_evaluation_seed = _nonnegative_integer(
        "evaluation_seed", evaluation_seed
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
        if dataset_metadata != checkpoint_training["dataset"]:
            raise ValueError(
                "continued training Dataset must match the checkpoint Dataset"
            )
    demonstration_data = (
        None
        if cloning is None
        else load_demonstrations(training_dataset, env=env)
    )

    directory = Path(output)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"training output is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to create training output in non-empty directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)

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
    if resolved_evaluate_every is not None:
        evaluation_recorder = _TrainingEvaluationRecorder(
            backend=backend,
            model=model,
            env=evaluation_env,
            checkpoint_env=env,
            evaluate_every=resolved_evaluate_every,
            seed=resolved_evaluation_seed,
            best_checkpoint=directory / "best" / "model.zip",
            initial_steps=initial_steps,
            checkpoint_state=checkpoint_state,
        )
        evaluation_recorder.start()

    def record_step(step: TrainingStep) -> None:
        cumulative = TrainingStep(
            step=initial_steps + step.step,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
        )
        curve_recorder.on_step(cumulative)
        if evaluation_recorder is not None:
            evaluation_recorder.on_step(cumulative)

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
    curve = curve_recorder.payload()
    checkpoint = directory / "model.zip"
    save_checkpoint(
        backend,
        model,
        checkpoint,
        env=env,
        training=checkpoint_state(total_steps),
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
    evaluation_history_path = None
    best_checkpoint = None
    best_tracking_figure = None
    if evaluation_recorder is not None:
        evaluation_history = evaluation_recorder.payload()
        evaluation_history_path = write_json(
            directory / "evaluation_history.json",
            evaluation_history,
        )
        best_checkpoint = directory / "best" / "model.zip"
        if not best_checkpoint.is_file():
            raise FileNotFoundError(
                f"training evaluation did not create checkpoint: {best_checkpoint}"
            )
        best_policy = load_policy(best_checkpoint, env=evaluation_env)
        best_evaluation = evaluate(
            env=evaluation_env,
            policy=best_policy,
            seeds=[resolved_evaluation_seed],
        )
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
                    "best policy - fixed training case"
                ),
            ),
            encoding="utf-8",
        )
        evaluation_metadata = {
            "environment": environment_metadata(evaluation_env),
            "evaluate_every": resolved_evaluate_every,
            "seed": resolved_evaluation_seed,
            "ranking_metric": evaluation_history["ranking_metric"],
            "best_step": evaluation_history["best_step"],
            "best_value": evaluation_history["best_value"],
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
    curve_path = write_json(directory / "training_curve.json", curve)
    curve_figure = plot_training_curve(
        curve,
        output=directory / "training_curve.svg",
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
        raise ValueError("evaluate_every requires evaluation_env")
    if evaluation_env is training_env:
        raise ValueError("evaluation_env must be separate from the training env")
    if evaluation_env.unwrapped.benchmark is not None:
        raise ValueError("training evaluation does not accept a Benchmark env")
    training_metadata = environment_metadata(training_env)
    evaluation_metadata = environment_metadata(evaluation_env)
    if evaluation_metadata["randomize"]:
        raise ValueError("training evaluation env must not randomize episodes")
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
        seed,
        best_checkpoint,
        initial_steps,
        checkpoint_state,
    ):
        self._backend = backend
        self._model = model
        self._env = env
        self._checkpoint_env = checkpoint_env
        self._evaluate_every = evaluate_every
        self._seed = seed
        self._best_checkpoint = best_checkpoint
        self._initial_steps = initial_steps
        self._checkpoint_state = checkpoint_state
        self._records = []
        self._best_step = None
        self._best_value = None
        self._best_index = None
        self._ranking_metric = None
        self._direction = None

    def start(self):
        self._evaluate(self._initial_steps)

    def on_step(self, event: TrainingStep):
        if event.step % self._evaluate_every == 0:
            self._evaluate(event.step)

    def finish(self, actual_steps):
        if not self._records or self._records[-1]["step"] != actual_steps:
            self._evaluate(actual_steps)

    def _evaluate(self, step):
        policy = self._backend.policy(
            self._model,
            checkpoint=self._best_checkpoint,
        )
        result = evaluate(env=self._env, policy=policy, seeds=[self._seed])
        ranking = result["ranking_metrics"][0]
        metric = ranking["name"]
        direction = ranking["direction"]
        if self._ranking_metric is None:
            self._ranking_metric = metric
            self._direction = direction
        elif metric != self._ranking_metric or direction != self._direction:
            raise ValueError("training evaluation ranking metric changed")
        value = float(result["aggregate"][metric]["median"])
        episode = result["episodes"][0]
        record = {
            "step": int(step),
            "value": value,
            "episode_return": float(episode["return"]),
            "episode_length": int(episode["length"]),
            "terminated": bool(episode["terminated"]),
            "truncated": bool(episode["truncated"]),
            "metrics": dict(episode["metrics"]),
        }
        self._records.append(record)
        improved = _is_better_training_evaluation(
            record,
            None if self._best_step is None else self._records[self._best_index],
            direction,
        )
        if improved:
            self._best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                self._backend,
                self._model,
                self._best_checkpoint,
                env=self._checkpoint_env,
                training=self._checkpoint_state(step),
                overwrite=True,
            )
            self._best_step = int(step)
            self._best_value = value
            self._best_index = len(self._records) - 1

    def payload(self):
        if not self._records or self._best_step is None:
            raise RuntimeError("training evaluation produced no records")
        return {
            "schema_version": TRAINING_EVALUATION_SCHEMA_VERSION,
            "evaluate_every": int(self._evaluate_every),
            "seed": int(self._seed),
            "ranking_metric": {
                "name": self._ranking_metric,
                "direction": self._direction,
            },
            "selection_order": [
                {"name": "safe_completion", "direction": "maximize"},
                {"name": "episode_length", "direction": "maximize"},
                {
                    "name": self._ranking_metric,
                    "direction": self._direction,
                },
            ],
            "best_step": int(self._best_step),
            "best_value": float(self._best_value),
            "records": list(self._records),
        }


def _is_better_training_evaluation(candidate, best, metric_direction):
    if best is None:
        return True
    candidate_completion = not candidate["terminated"]
    best_completion = not best["terminated"]
    if candidate_completion != best_completion:
        return candidate_completion
    if candidate["episode_length"] != best["episode_length"]:
        return candidate["episode_length"] > best["episode_length"]
    if metric_direction == "minimize":
        return candidate["value"] < best["value"]
    if metric_direction == "maximize":
        return candidate["value"] > best["value"]
    raise ValueError("training evaluation metric direction must be minimize or maximize")


class _TrainingCurveRecorder:
    def __init__(self, record_every, *, initial_steps=0):
        self._record_every = record_every
        self._initial_steps = initial_steps
        self._window_start = initial_steps
        self._window_rewards = []
        self._window_completed = 0
        self._window_terminated = 0
        self._window_truncated = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._records = []
        self._episodes = []
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


__all__ = ["TRAINING_SCHEMA_VERSION", "load_policy", "train"]

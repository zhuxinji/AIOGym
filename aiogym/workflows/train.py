"""Thin optional Stable-Baselines3 training workflow."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiogym.controllers.policies import SB3CheckpointPolicy
from aiogym.core import file_sha256, make_env, resolve_condition_alias, write_json

from .artifacts import write_run_bundle
from .evaluate import evaluate


TRAIN_SCHEMA_VERSION = "aiogym.train.v1"
ALGORITHMS = ("ddpg", "ppo", "sac", "td3")


@dataclass(frozen=True)
class TrainConfig:
    task: str
    algorithm: str
    steps: int
    output: Path
    plant: Any = None
    condition: Any = None
    seed: int = 0
    eval_seeds: tuple[int, ...] = (100,)
    algorithm_kwargs: Mapping[str, Any] = field(default_factory=dict)
    eval_max_steps: int | None = None

    def __post_init__(self):
        algorithm = str(self.algorithm).lower()
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
        if isinstance(self.steps, bool) or int(self.steps) <= 0:
            raise ValueError("steps must be a positive integer")
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.eval_seeds:
            raise ValueError("eval_seeds must not be empty")
        resolved_eval_seeds = []
        for value in self.eval_seeds:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("eval_seeds must contain non-negative integers")
            resolved_eval_seeds.append(int(value))
        if len(set(resolved_eval_seeds)) != len(resolved_eval_seeds):
            raise ValueError("eval_seeds must not contain duplicates")
        if self.eval_max_steps is not None and (
            isinstance(self.eval_max_steps, bool) or int(self.eval_max_steps) <= 0
        ):
            raise ValueError("eval_max_steps must be a positive integer")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "eval_seeds", tuple(resolved_eval_seeds))
        object.__setattr__(
            self,
            "algorithm_kwargs",
            copy.deepcopy(dict(self.algorithm_kwargs)),
        )


def train(
    *,
    task: str,
    algorithm: str,
    steps: int,
    output: str | Path,
    plant=None,
    condition=None,
    preset: str | None = None,
    seed: int = 0,
    eval_seeds: Sequence[int] = (100,),
    algorithm_kwargs: Mapping[str, Any] | None = None,
    eval_max_steps: int | None = None,
    overwrite: bool = False,
):
    config = TrainConfig(
        task=task,
        algorithm=algorithm,
        steps=steps,
        output=Path(output),
        plant=plant,
        condition=resolve_condition_alias(condition, preset),
        seed=seed,
        eval_seeds=tuple(eval_seeds),
        algorithm_kwargs=dict(algorithm_kwargs or {}),
        eval_max_steps=eval_max_steps,
    )
    if config.output.exists() and any(config.output.iterdir()) and not overwrite:
        raise FileExistsError(f"training output already exists: {config.output}")
    config.output.mkdir(parents=True, exist_ok=True)
    model_directory = config.output / "model"
    model_directory.mkdir(parents=True, exist_ok=True)
    env = make_env(config.task, plant=config.plant, condition=config.condition)
    environment_contract = env.identity.as_dict()
    algorithm_class = _algorithm_class(config.algorithm)
    kwargs = _algorithm_defaults(config.algorithm, config.steps)
    kwargs.update(copy.deepcopy(dict(config.algorithm_kwargs)))
    resolved_kwargs = copy.deepcopy(kwargs)
    model = algorithm_class(
        kwargs.pop("policy", "MlpPolicy"),
        env,
        seed=config.seed,
        **kwargs,
    )
    try:
        model.learn(total_timesteps=config.steps)
        checkpoint_base = model_directory / "model"
        model.save(str(checkpoint_base))
    finally:
        env.close()
    checkpoint = checkpoint_base.with_suffix(".zip")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SB3 did not create checkpoint: {checkpoint}")
    contract = {
        **environment_contract,
        "algorithm": config.algorithm,
    }
    contract_path = write_json(model_directory / "contract.json", contract)
    policy = SB3CheckpointPolicy.load(
        checkpoint,
        algorithm=config.algorithm,
        device="cpu",
    )
    evaluation = evaluate(
        policy,
        task=config.task,
        plant=config.plant,
        condition=config.condition,
        seeds=config.eval_seeds,
        max_steps=config.eval_max_steps,
    )
    result = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "workflow": "train",
        "task_id": evaluation["task_id"],
        "task_hash": evaluation["task_hash"],
        "plant_id": evaluation["plant_id"],
        "plant_hash": evaluation["plant_hash"],
        "condition_id": evaluation["condition_id"],
        "condition_hash": evaluation["condition_hash"],
        "interface_hash": evaluation["interface_hash"],
        "env_hash": evaluation["env_hash"],
        "algorithm": config.algorithm,
        "steps": config.steps,
        "seed": config.seed,
        "seeds": list(config.eval_seeds),
        "algorithm_kwargs": resolved_kwargs,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "checkpoint_contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
            "contract": contract,
        },
        "evaluation": evaluation,
        "metrics": evaluation["aggregate"],
    }
    result["artifacts"] = write_run_bundle(
        result,
        config.output,
        workflow="train",
        report_markdown=render_training_report(result),
        overwrite=overwrite,
    )
    return result


def load_checkpoint(
    checkpoint: str | Path,
    *,
    algorithm: str,
    device: str = "auto",
):
    return SB3CheckpointPolicy.load(
        checkpoint,
        algorithm=algorithm,
        device=device,
    )


def _algorithm_class(algorithm):
    try:
        from stable_baselines3 import DDPG, PPO, SAC, TD3
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stable-Baselines3 is required for train(); install `aiogym[rl]`"
        ) from error
    return {"sac": SAC, "ppo": PPO, "td3": TD3, "ddpg": DDPG}[algorithm]


def _algorithm_defaults(algorithm, steps):
    if algorithm == "ppo":
        n_steps = min(2048, max(2, int(steps)))
        return {
            "n_steps": n_steps,
            "batch_size": min(64, n_steps),
            "n_epochs": 1,
            "verbose": 0,
            "device": "cpu",
        }
    return {
        "learning_starts": 0,
        "buffer_size": max(100, int(steps) + 1),
        "batch_size": min(64, max(2, int(steps))),
        "train_freq": 1,
        "gradient_steps": 1,
        "verbose": 0,
        "device": "cpu",
    }


def render_training_report(result):
    return "\n".join(
        (
            f"# Training: {result['task_id']}",
            "",
            f"Algorithm: `{result['algorithm']}`",
            "",
            f"Training steps: {result['steps']}",
            "",
            f"Checkpoint: `{result['checkpoint']['path']}`",
            "",
            "This run is a pipeline execution record; smoke budgets do not establish policy performance.",
        )
    )


__all__ = [
    "ALGORITHMS",
    "TRAIN_SCHEMA_VERSION",
    "TrainConfig",
    "load_checkpoint",
    "train",
]

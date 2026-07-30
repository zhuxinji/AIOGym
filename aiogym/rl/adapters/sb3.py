"""Stable adapter for SAC, TD3, and PPO through SB3."""
from __future__ import annotations

from ._delegating import (
    DelegatingTrainingAdapter,
    add_boolean_option,
    add_option,
)


class SB3TrainingAdapter(DelegatingTrainingAdapter):
    backend = "sb3"

    def _run_backend(self) -> None:
        from aiogym.rl.train_sb3 import _run_backend

        config = self.plan.config
        algorithm = dict(config.algorithm)
        replay = dict(config.replay)
        evaluation = dict(config.evaluation)
        output = dict(config.output)
        argv = [
            "--track",
            self.plan.track.id,
            "--algo",
            config.algorithm_id,
            "--steps",
            str(config.total_transitions),
            "--n-envs",
            str(config.n_envs),
            "--seed",
            str(config.training_seed),
            "--device",
            config.device,
            "--out-dir",
            str(self.plan.output_dir),
            "--name",
            self.plan.run_name,
            "--artifact-dir",
            str(self.plan.artifact_dir),
            "--eval-seed-list",
            ",".join(str(seed) for seed in config.validation_seeds),
            "--eval-episodes",
            str(len(config.validation_seeds)),
            "--learning-curve-episodes",
            str(len(config.validation_seeds)),
        ]
        mapping = (
            (algorithm, "learning_rate", "--learning-rate"),
            (algorithm, "gamma", "--gamma"),
            (algorithm, "batch_size", "--batch-size"),
            (algorithm, "rollout_vector_steps", "--train-freq"),
            (algorithm, "utd_ratio", "--utd-ratio"),
            (algorithm, "n_steps", "--ppo-n-steps"),
            (algorithm, "torch_threads", "--torch-threads"),
            (algorithm, "verbose", "--verbose"),
            (replay, "capacity", "--buffer-size"),
            (replay, "learning_starts", "--learning-starts"),
            (
                evaluation,
                "every_transitions",
                "--learning-curve-every",
            ),
        )
        for source, name, flag in mapping:
            add_option(argv, flag, source.get(name))
        add_option(argv, "--vec-env", algorithm.get("vector_backend"))
        add_option(argv, "--resume", config.resume_checkpoint)
        add_boolean_option(
            argv,
            "--save-rollout",
            output.get("save_rollout"),
        )
        if bool(output.get("onnx", False)):
            argv.append("--onnx")
        _run_backend(argv, prog="aiogym train")


__all__ = ["SB3TrainingAdapter"]

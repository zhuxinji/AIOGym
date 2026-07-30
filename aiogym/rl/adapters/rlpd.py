"""Stable adapter for dataset-to-online RLPD."""
from __future__ import annotations

from ._delegating import (
    DelegatingTrainingAdapter,
    add_boolean_option,
    add_option,
)


class RLPDTrainingAdapter(DelegatingTrainingAdapter):
    backend = "rlpd"

    def _run_backend(self) -> None:
        from aiogym.rl.train_rlpd import _run_backend

        config = self.plan.config
        algorithm = dict(config.algorithm)
        replay = dict(config.replay)
        evaluation = dict(config.evaluation)
        output = dict(config.output)
        argv = [
            "--track",
            self.plan.track.id,
            "--dataset",
            str(self.plan.dataset_path),
            "--dataset-id",
            self.plan.dataset_id,
            "--online-steps",
            str(config.total_transitions),
            "--n-envs",
            str(config.n_envs),
            "--seed",
            str(config.training_seed),
            "--device",
            config.device,
            "--out",
            str(self.plan.policy_path.with_suffix("")),
            "--artifact-dir",
            str(self.plan.artifact_dir),
            "--validation-episodes",
            str(len(config.validation_seeds)),
            "--validation-seed-list",
            ",".join(str(seed) for seed in config.validation_seeds),
        ]
        mapping = (
            (algorithm, "bc_steps", "--bc-steps"),
            (algorithm, "pretrain_updates", "--pretrain-updates"),
            (algorithm, "utd_ratio", "--utd"),
            (algorithm, "n_critics", "--n-critics"),
            (algorithm, "batch_size", "--batch-size"),
            (algorithm, "offline_fraction", "--offline-fraction"),
            (replay, "capacity", "--online-capacity"),
            (evaluation, "every_transitions", "--eval-every"),
        )
        for source, name, flag in mapping:
            add_option(argv, flag, source.get(name))
        add_option(argv, "--resume", config.resume_checkpoint)
        add_boolean_option(
            argv,
            "--save-rollout",
            output.get("save_rollout"),
        )
        if (
            "offline_fraction" in algorithm
            and float(algorithm["offline_fraction"]) != 0.5
        ):
            argv.append("--allow-offline-ratio-variant")
        _run_backend(argv, prog="aiogym train")


__all__ = ["RLPDTrainingAdapter"]

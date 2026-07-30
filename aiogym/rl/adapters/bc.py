"""Stable adapter for Dataset v2 behavior-cloning sanity training."""
from __future__ import annotations

from ._delegating import DelegatingTrainingAdapter, add_option


class BCTrainingAdapter(DelegatingTrainingAdapter):
    backend = "bc"

    def _run_backend(self) -> None:
        from aiogym.rl.train_offline import _run_backend

        config = self.plan.config
        algorithm = dict(config.algorithm)
        argv = [
            "--dataset",
            str(self.plan.dataset_path),
            "--dataset-id",
            self.plan.dataset_id,
            "--steps",
            str(config.total_transitions),
            "--device",
            config.device,
            "--seed",
            str(config.training_seed),
            "--out",
            str(self.plan.policy_path),
        ]
        for name, flag in (
            ("batch_size", "--batch-size"),
            ("hidden", "--hidden"),
            ("learning_rate", "--learning-rate"),
        ):
            add_option(argv, flag, algorithm.get(name))
        _run_backend(argv)


__all__ = ["BCTrainingAdapter"]

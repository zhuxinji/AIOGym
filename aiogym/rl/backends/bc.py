"""Direct Dataset v2 behavior-cloning backend."""
from __future__ import annotations

from pathlib import Path
from time import monotonic

from ..behavior_cloning import BehaviorCloningTrainer
from ..dataset_replay import DatasetReplay
from . import BackendResult


def run_bc(plan) -> BackendResult:
    """Fit behavior cloning and return its selected native checkpoint."""

    started_at = monotonic()
    config = plan.config
    algorithm = dict(config.algorithm)

    replay = DatasetReplay(
        plan.dataset_path,
        seed=config.training_seed,
        stratify=True,
        verify_checksums=True,
    )
    if plan.dataset_id is not None and replay.dataset_id != plan.dataset_id:
        raise ValueError(
            f"dataset_id {plan.dataset_id!r} does not match "
            f"manifest {replay.dataset_id!r}"
        )
    if config.algorithm_id != "bc":
        raise ValueError("resolved config algorithm_id must be 'bc'")
    if config.dataset_id != replay.dataset_id:
        raise ValueError(
            "resolved config dataset_id does not match the dataset"
        )
    trainer = BehaviorCloningTrainer(
        replay,
        hidden=int(algorithm["hidden"]),
        learning_rate=float(algorithm["learning_rate"]),
        device=config.device,
        seed=config.training_seed,
    )
    report = trainer.fit(
        steps=config.total_transitions,
        batch_size=int(algorithm["batch_size"]),
    )
    report["kind"] = "behavior_cloning_sanity"
    report["action_contract"] = "normalized[-1,1]"
    report["training_config"] = config.as_dict()
    report["training_config_hash"] = config.config_hash
    import torch

    target = Path(plan.policy_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.checkpoint_payload(report), target)
    return BackendResult(
        algorithm_id=config.algorithm_id,
        policy_path=target,
        final_step=int(report["steps"]),
        checkpoint_selection="final-fixed-dataset",
        training_metadata={
            "dataset_id": plan.dataset_id,
            "dataset_hash": plan.dataset_hash,
            "optimizer_steps": int(report["steps"]),
            "batch_size": int(algorithm["batch_size"]),
            "hidden": int(algorithm["hidden"]),
            "learning_rate": float(algorithm["learning_rate"]),
            "initial_loss": float(report["initial_mse"]),
            "final_loss": float(report["final_mse"]),
            "action_contract": "normalized[-1,1]",
        },
        runtime={"seconds": monotonic() - started_at},
    )


__all__ = ["run_bc"]

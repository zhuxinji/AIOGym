"""Internal Dataset v2 behavior-cloning kernel for the unified adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .behavior_cloning import BehaviorCloningTrainer
from .dataset_replay import DatasetReplay


def _run_backend(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    replay = DatasetReplay(
        args.dataset,
        seed=args.seed,
        stratify=True,
        verify_checksums=True,
    )
    if args.dataset_id is not None and replay.dataset_id != args.dataset_id:
        parser.error(
            f"--dataset-id {args.dataset_id!r} does not match "
            f"manifest {replay.dataset_id!r}"
        )
    trainer = BehaviorCloningTrainer(
        replay,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    report = trainer.fit(steps=args.steps, batch_size=args.batch_size)
    report["kind"] = "behavior_cloning_sanity"
    report["action_contract"] = "normalized[-1,1]"
    if args.out:
        import torch

        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trainer": trainer.state_dict(),
                "report": report,
            },
            target,
        )
    print(json.dumps(report, indent=2, sort_keys=True))

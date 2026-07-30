from __future__ import annotations

from pathlib import Path

from aiogym.benchmarks import load_track
from aiogym.rl.validation import ValidationEpisodePlan


ROOT = Path(__file__).resolve().parents[2]


def test_train_and_hpo_modules_do_not_access_test_split():
    offenders = []
    for path in (
        ROOT / "aiogym" / "rl" / "train_sb3.py",
        ROOT / "aiogym" / "rl" / "train_rlpd.py",
        ROOT / "aiogym" / "rl" / "hpo.py",
        ROOT / "aiogym" / "rl" / "training_config.py",
    ):
        source = path.read_text(encoding="utf-8")
        for forbidden in (
            'split="test"',
            "split='test'",
            'resolved_cases("test")',
            "resolved_cases('test')",
            "test_seed_namespace",
        ):
            if forbidden in source:
                offenders.append(f"{path.name}: {forbidden}")
    assert offenders == []


def test_validation_plan_is_fixed_and_hash_stable():
    track = load_track("quadruple-regulation-generalist-v1")
    first = ValidationEpisodePlan(track, base_seeds=[100, 101])
    second = ValidationEpisodePlan(track, base_seeds=[100, 101])
    assert first.plan_hash == second.plan_hash
    assert first.metadata() == second.metadata()

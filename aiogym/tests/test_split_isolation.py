from __future__ import annotations

from pathlib import Path

import pytest

import aiogym.benchmarks as benchmarks
import aiogym.benchmarks.evaluation as benchmark_evaluation
from aiogym.benchmarks import evaluate_policy_on_track, load_track
from aiogym.cli.evaluate import build_parser as build_evaluate_parser
from aiogym.rl.validation import ValidationEpisodePlan


ROOT = Path(__file__).resolve().parents[2]


def test_train_and_hpo_modules_do_not_access_test_split():
    offenders = []
    for path in (
        ROOT / "aiogym" / "rl" / "backends" / "sb3.py",
        ROOT / "aiogym" / "rl" / "backends" / "rlpd.py",
        ROOT / "aiogym" / "rl" / "hpo.py",
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


def test_public_track_evaluator_is_validation_only(monkeypatch):
    assert not hasattr(benchmarks, "_evaluate_policy_on_track_split")
    calls = []

    def evaluate_split(controller, track, **kwargs):
        calls.append((controller, track, kwargs))
        return {"split": kwargs["split"]}

    monkeypatch.setattr(
        benchmark_evaluation,
        "_evaluate_policy_on_track_split",
        evaluate_split,
    )
    controller = object()
    result = evaluate_policy_on_track(
        controller,
        "quadruple-regulation-generalist-v1",
        base_seeds=(101,),
        include_episodes=False,
    )

    assert result == {"split": "validation"}
    assert calls[0][0] is controller
    assert calls[0][2]["split"] == "validation"
    with pytest.raises(TypeError, match="unexpected keyword argument 'split'"):
        evaluate_policy_on_track(
            controller,
            "quadruple-regulation-generalist-v1",
            split="test",
        )


def test_private_track_evaluator_rejects_non_benchmark_splits():
    with pytest.raises(ValueError, match="validation.*test"):
        benchmark_evaluation._evaluate_policy_on_track_split(
            object(),
            "quadruple-regulation-generalist-v1",
            split="training",
            base_seeds=(101,),
            include_episodes=False,
        )


@pytest.mark.parametrize("split", ("validation", "test"))
def test_evaluate_cli_has_no_split_option(split):
    parser = build_evaluate_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--split" not in options
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "unused.zip",
                "--track",
                "quadruple-regulation-generalist-v1",
                "--algorithm",
                "sac",
                "--split",
                split,
            ]
        )


def test_validation_plan_is_fixed_and_hash_stable():
    track = load_track("quadruple-regulation-generalist-v1")
    first = ValidationEpisodePlan(track, base_seeds=[100, 101])
    second = ValidationEpisodePlan(track, base_seeds=[100, 101])
    assert first.plan_hash == second.plan_hash
    assert first.metadata() == second.metadata()

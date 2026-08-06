from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aiogym.scenarios  # noqa: F401
from aiogym.controllers.base import make_controller
from aiogym.core import make_env
from aiogym.workflows import evaluate, load_plant


DESIGN = (
    Path(__file__).resolve().parents[2]
    / "aiogym/scenarios/three_tank/default-design-v1.json"
)


def test_regulation_evaluate_aggregates_explicit_seeds():
    result = evaluate(
        "pid",
        task="quadruple/regulation",
        preset="minimum-phase",
        seeds=(3, 4),
        max_steps=4,
    )
    assert result["primary_metric"] == "tracking_iae"
    assert result["metric_direction"] == "minimize"
    assert [row["seed"] for row in result["episodes"]] == [3, 4]
    assert result["aggregate"]["return"]["std"] == pytest.approx(0.0)
    assert result["aggregate"]["constraint_violations"]["mean"] == 0.0
    assert result["aggregate"]["termination"]["mean"] == 0.0
    assert "tracking_ise" in result["aggregate"]


def test_economic_and_regulation_are_distinct_tasks_on_same_plant():
    regulation = evaluate(
        "pid",
        task="cascade/regulation",
        preset="continuous-benchmark",
        seeds=(0,),
        max_steps=3,
    )
    economic = evaluate(
        "pid",
        task="cascade/economic",
        preset="continuous-benchmark",
        seeds=(0,),
        max_steps=3,
    )
    assert regulation["plant_hash"] == economic["plant_hash"]
    assert regulation["task_hash"] != economic["task_hash"]
    assert "tracking_iae" in regulation["aggregate"]
    assert "economic_objective" in economic["aggregate"]
    assert "tracking_iae" not in economic["aggregate"]


def test_bound_policy_evaluate_keeps_plant_and_task_identity():
    plant = load_plant(DESIGN)
    env = make_env("three_tank/regulation", plant=plant)
    try:
        policy = make_controller("pid", env=env)
        result = evaluate(
            policy,
            task="three_tank/regulation",
            plant=plant,
            seeds=(8, 9),
            max_steps=3,
        )
        assert result["plant_hash"] == plant.plant_hash
        assert result["policy"]["kind"] == "matrix_pid"
        assert np.isfinite(result["aggregate"]["energy"]["mean"])
    finally:
        env.close()


def test_evaluate_artifacts_and_seed_validation(tmp_path):
    output = tmp_path / "evaluation"
    result = evaluate(
        "hold",
        task="quadruple/regulation",
        preset="minimum-phase",
        seeds=(1,),
        max_steps=2,
        output=output,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "evaluate"
    assert manifest["plant_hash"] == result["plant_hash"]
    assert manifest["seeds"] == [1]
    assert manifest["policy"]["id"] == "hold"
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()
    with pytest.raises(FileExistsError):
        evaluate(
            "hold",
            task="quadruple/regulation",
            preset="minimum-phase",
            seeds=(1,),
            max_steps=2,
            output=output,
        )
    with pytest.raises(ValueError, match="duplicates"):
        evaluate("hold", task="quadruple/regulation", seeds=(1, 1), max_steps=1)
    with pytest.raises(TypeError, match="non-negative integers"):
        evaluate("hold", task="quadruple/regulation", seeds=(True,), max_steps=1)


def test_evaluate_source_has_no_ranking_protocol_dependencies():
    source = (
        Path(__file__).resolve().parents[2] / "aiogym/workflows/evaluate.py"
    ).read_text(encoding="utf-8")
    for retired in ("Track", "Anchor", "Scorecard", "Goal registry", "ranking"):
        assert retired not in source

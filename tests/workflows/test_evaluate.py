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
    / "aiogym/scenarios/three_tank/plants/lab-three-tank-v1.json"
)


def test_regulation_evaluate_aggregates_explicit_seeds():
    result = evaluate(
        "pid",
        task="quadruple/regulation",
        condition="minimum-phase",
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
        task="three_tank/regulation",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
        seeds=(0,),
        max_steps=3,
    )
    economic = evaluate(
        "pid",
        task="three_tank/economic",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
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
        condition="minimum-phase",
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
            condition="minimum-phase",
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


class _ContractPolicy:
    def __init__(self, action_dim, contract):
        self.action_dim = action_dim
        self.contract = contract

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del observation, context
        return np.zeros(self.action_dim, dtype=np.float32)

    def metadata(self):
        return {
            "id": "contract-test",
            "training_contract": self.contract,
        }


def test_checkpoint_interface_rejection_and_condition_transfer_flags():
    source = make_env(
        "three_tank/regulation", plant="recirculating-h1-v1"
    )
    try:
        contract = source.identity.as_dict()
        policy = _ContractPolicy(4, contract)
    finally:
        source.close()
    with pytest.raises(ValueError, match="interface_hash"):
        evaluate(
            policy,
            task="three_tank/regulation",
            plant="lab-three-tank-v1",
            seeds=(0,),
            max_steps=1,
        )

    base = load_plant("recirculating-h1-v1").conditions["commissioning"]
    changed = base.as_dict(include_hash=False)
    changed["id"] = "transfer-condition"
    changed["reference"] = [*base.reference[:3], 31.0, *base.reference[4:]]
    with pytest.raises(ValueError, match="allow_condition_transfer"):
        evaluate(
            policy,
            task="three_tank/regulation",
            plant="recirculating-h1-v1",
            condition=changed,
            seeds=(0,),
            max_steps=1,
        )
    result = evaluate(
        policy,
        task="three_tank/regulation",
        plant="recirculating-h1-v1",
        condition=changed,
        seeds=(0,),
        max_steps=1,
        allow_condition_transfer=True,
    )
    assert result["transfer_flags"] == {
        "is_transfer": True,
        "plant_changed": False,
        "condition_changed": True,
    }

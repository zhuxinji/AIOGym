from __future__ import annotations

import copy
import json

import pytest

import aiogym
from aiogym.tests._env import make_test_env
from aiogym.evaluation import goal_spec
from aiogym.evaluation.metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    SCORECARD_GROUPS,
    metric_definitions,
)
from aiogym.rewards import list_reward_specs


def _evaluate(reward_spec: str):
    env = make_test_env(
        "cstr",
        reward_spec=reward_spec,
        episode_steps=5,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
        noise=False,
    )
    controller = aiogym.make_controller("pid", scenario="cstr")
    return aiogym.evaluate_controller(controller, env, seed_list=[17])


def test_goal_spec_only_accepts_regulation_and_economic():
    assert goal_spec("regulation").primary_metric == "regulation_cost"
    assert goal_spec("economic").primary_metric == "profit"
    with pytest.raises(ValueError, match="economic, regulation"):
        goal_spec("tracking")
    with pytest.raises(ValueError, match="not a cross-reward ranking metric"):
        goal_spec("regulation", primary_metric="return")


def test_same_trajectory_has_same_scorecard_under_all_reward_specs():
    specs = list_reward_specs()
    results = [_evaluate(spec) for spec in specs]
    baseline = results[0]["scorecard"]
    assert all(result["scorecard"] == baseline for result in results[1:])
    assert len({result["return"] for result in results}) > 1


def test_all_safety_metrics_exist_for_regulation_and_economic():
    for reward_spec in ("regulation-v1", "economic-v1"):
        result = _evaluate(reward_spec)
        safety = result["scorecard"]["safety"]
        assert set(safety) == set(SCORECARD_GROUPS["safety"])
        assert all(metric in result for metric in SCORECARD_GROUPS["safety"])
        assert result["safety_gate"]["mode"] == "ordinary"
        assert result["ranking_eligible"] is True


def test_return_is_tagged_with_reward_spec_id():
    result = _evaluate("regulation-v1")
    assert result["reward_spec_id"] == "regulation-v1"
    assert result["return_metadata"] == {
        "reward_spec_id": "regulation-v1",
        "comparable_across_reward_specs": False,
        "description": "sum of the environment training reward",
    }


def test_metric_catalog_is_grouped_not_objective_filtered():
    assert set(SCORECARD_GROUPS) == {
        "regulation",
        "economics",
        "safety",
        "controller",
    }
    assert metric_definitions() == dict(METRIC_DEFINITIONS)


def test_legacy_artifact_reader_still_loads_v4(tmp_path):
    legacy = {
        "schema_version": "aiogym.evaluation.v4",
        "controller_name": "legacy-controller",
        "objective": "tracking",
        "metric": "normalized_score",
        "normalized_score": 91.0,
        "normalized_score_std": 1.0,
        "objective_status": "met",
        "objective_acceptance": {"status": "met", "checks": []},
        "return": -2.0,
    }
    source = copy.deepcopy(legacy)
    path = tmp_path / "evaluation-v4.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    from aiogym.evaluation.legacy_artifacts import (
        load_legacy_evaluation_artifact,
    )

    migrated = load_legacy_evaluation_artifact(path)

    assert source == legacy
    assert migrated["schema_version"] == "aiogym.evaluation.v4"
    assert migrated["metric"] == "normalized_score"
    assert migrated["normalized_score"] == 91.0
    assert migrated["acceptance_status"] == "met"
    assert migrated["goal"] == "regulation"
    assert "objective_status" not in migrated

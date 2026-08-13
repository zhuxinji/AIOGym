from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from aiogym import compare_policies, evaluate, make_controller, make_env


def test_evaluate_preserves_seed_order_and_aggregates_metrics():
    env = make_env("quadruple")
    try:
        result = evaluate(
            env=env,
            policy="pid",
            seeds=(4, 3),
            max_steps=4,
        )
        observation, _ = env.reset(seed=20)
        assert observation.shape == env.observation_space.shape
    finally:
        env.close()

    assert result["schema_version"] == "aiogym.evaluation.v3"
    assert result["ranking_metrics"] == [
        {"name": "tracking_iae", "direction": "minimize"}
    ]
    assert [row["seed"] for row in result["episodes"]] == [4, 3]
    assert [row["length"] for row in result["episodes"]] == [4, 4]
    assert set(result["aggregate"]["episode_return"]) == {
        "mean",
        "std",
        "median",
        "mad",
        "min",
        "max",
    }
    returns = np.asarray([row["return"] for row in result["episodes"]])
    assert result["aggregate"]["episode_return"] == pytest.approx(
        {
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "median": float(np.median(returns)),
            "mad": float(np.median(np.abs(returns - np.median(returns)))),
            "min": float(returns.min()),
            "max": float(returns.max()),
        }
    )
    assert result["aggregate"]["episode_length"] == {
        "mean": 4.0,
        "std": 0.0,
        "median": 4.0,
        "mad": 0.0,
        "min": 4.0,
        "max": 4.0,
    }
    assert "tracking_ise" in result["aggregate"]
    assert result["return_distribution"] == pytest.approx(returns.tolist())
    assert result["episodes"][0]["trajectory"]["physical_time"] == pytest.approx(
        [1.0, 2.0, 3.0, 4.0]
    )
    assert len(result["episodes"][0]["trajectory"]["true_state"]) == 4
    assert result["trajectory_summary"]["output"]["samples"] == [2, 2, 2, 2]
    assert result["trajectory_schema"]["output"][0]["name"] == "lower_tank_1_level"
    assert result["trajectory_schema"]["output"][0]["low"] == 0.0
    assert result["trajectory_schema"]["output"][0]["high"] == 20.0


def test_reward_metric_sets_remain_distinct():
    regulation_env = make_env("three_tank", reward="regulation")
    economic_env = make_env("three_tank", reward="economic")
    try:
        regulation = evaluate(
            env=regulation_env,
            policy="pid",
            seeds=(0,),
            max_steps=3,
        )
        economic = evaluate(
            env=economic_env,
            policy="pid",
            seeds=(0,),
            max_steps=3,
        )
    finally:
        regulation_env.close()
        economic_env.close()
    assert "tracking_iae" in regulation["aggregate"]
    assert "economic_objective" in economic["aggregate"]
    assert "tracking_iae" not in economic["aggregate"]


def test_tank3_reward_reports_diagnostic_metrics():
    env = make_env(
        "three_tank",
        reward="tank3-regulation",
    )
    try:
        result = evaluate(env=env, policy="hold", seeds=(0,), max_steps=3)
    finally:
        env.close()
    assert result["ranking_metrics"] == [
        {"name": "tank3_tracking_iae", "direction": "minimize"}
    ]
    for metric in (
        "tank3_level_iae",
        "tank3_temperature_iae",
        "upstream_level_iae",
        "upstream_level_max_error",
        "final_tank3_level_error_m",
        "action_slew_violation_count",
        "heater_total_variation",
    ):
        assert metric in result["aggregate"]


def test_evaluate_writes_one_json_and_rejects_overwrite(tmp_path):
    output = tmp_path / "evaluation.json"
    env = make_env("quadruple")
    try:
        result = evaluate(
            env=env,
            policy="hold",
            seeds=(1,),
            max_steps=2,
            output=output,
        )
        with pytest.raises(FileExistsError):
            evaluate(
                env=env,
                policy="hold",
                seeds=(1,),
                max_steps=2,
                output=output,
            )
    finally:
        env.close()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert [path.name for path in tmp_path.iterdir()] == ["evaluation.json"]


@pytest.mark.parametrize(
    ("seeds", "error"),
    [
        ((), ValueError),
        ((1, 1), ValueError),
        ((True,), TypeError),
        ((-1,), ValueError),
    ],
)
def test_evaluate_validates_seeds(seeds, error):
    env = make_env("quadruple")
    try:
        with pytest.raises(error):
            evaluate(env=env, policy="hold", seeds=seeds, max_steps=1)
    finally:
        env.close()


def test_compare_matches_individual_evaluation_and_minimize_ordering(tmp_path):
    env = make_env("quadruple")
    try:
        pid = make_controller("pid", env=env)
        single = evaluate(env=env, policy=pid, seeds=(3, 4), max_steps=3)
        comparison = compare_policies(
            env=env,
            policies={"pid": pid, "hold": "hold"},
            seeds=(3, 4),
            max_steps=3,
            output=tmp_path / "comparison",
        )
    finally:
        env.close()
    assert comparison["seeds"] == [3, 4]
    assert comparison["evaluations"]["pid"] == single
    medians = {
        label: result["aggregate"]["tracking_iae"]["median"]
        for label, result in comparison["evaluations"].items()
    }
    assert comparison["ordering"] == sorted(
        medians, key=lambda label: (medians[label], label)
    )


def test_compare_uses_maximize_direction_and_writes_json_and_svg(tmp_path):
    output = tmp_path / "comparison"
    env = make_env("three_tank", reward="economic")
    try:
        result = compare_policies(
            env=env,
            policies={"pid": "pid", "hold": "hold"},
            seeds=(0, 1),
            max_steps=2,
            output=output,
        )
    finally:
        env.close()
    medians = {
        label: evaluation["aggregate"]["economic_objective"]["median"]
        for label, evaluation in result["evaluations"].items()
    }
    assert result["ranking_metrics"] == [
        {"name": "economic_objective", "direction": "maximize"}
    ]
    assert result["ordering"] == sorted(
        medians, key=lambda label: (-medians[label], label)
    )
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == result
    svg_path = output / "comparison.svg"
    ET.parse(svg_path)
    svg = svg_path.read_text(encoding="utf-8")
    return_svg = svg.split(">Cumulative return by policy</text>", 1)[1]
    assert ">pid</text>" in return_svg
    assert ">hold</text>" in return_svg
    assert ">episode return (higher is better)</text>" in return_svg
    assert "circles: evaluation seeds; diamond: median" in return_svg
    series_svg = svg.split(">Cumulative return by policy</text>", 1)[0]
    x_axis_starts = re.findall(
        r'<text class="tick" text-anchor="middle" x="76\.00" y="[^"]+">([^<]+)</text>',
        series_svg,
    )
    assert x_axis_starts
    assert set(x_axis_starts) == {"0"}
    first_level = svg.split(">Output: tank_1_level [m]</text>", 1)[1]
    first_level = first_level.split('<text class="panel-title"', 1)[0]
    level_ticks = re.findall(
        r'<text class="tick" text-anchor="end" x="68\.0" y="[^"]+">([^<]+)</text>',
        first_level,
    )
    assert level_ticks == ["0", "0.133", "0.267", "0.4"]
    first_action = svg.split(">Applied action: pump_P101 [fraction]</text>", 1)[1]
    first_action = first_action.split('<text class="panel-title"', 1)[0]
    action_ticks = re.findall(
        r'<text class="tick" text-anchor="end" x="68\.0" y="[^"]+">([^<]+)</text>',
        first_action,
    )
    assert action_ticks == ["0", "0.333", "0.667", "1"]
    assert sorted(path.name for path in output.iterdir()) == [
        "comparison.json",
        "comparison.svg",
    ]

    env = make_env("three_tank", reward="economic")
    try:
        with pytest.raises(FileExistsError, match="non-empty directory"):
            compare_policies(
                env=env,
                policies={"pid": "pid", "hold": "hold"},
                seeds=(0,),
                max_steps=1,
                output=output,
            )
    finally:
        env.close()


def test_compare_uses_benchmark_declared_lexicographic_ranking(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    env = make_env("quadruple", benchmark="disturbance-rejection")
    try:
        result = compare_policies(
            env=env,
            policies={"pid": "pid", "hold": "hold"},
            seeds=(0, 1),
            max_steps=2,
        )
    finally:
        env.close()
    assert result["ranking_metrics"] == [
        {"name": "unsafe_rate", "direction": "minimize"},
        {"name": "disturbance_iae", "direction": "minimize"},
        {"name": "recovery_time", "direction": "minimize"},
    ]
    output = tmp_path / "runs" / "quadruple" / "disturbance-rejection"
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == result
    ET.parse(output / "comparison.svg")

    env = make_env("quadruple", benchmark="disturbance-rejection")
    try:
        replacement = compare_policies(
            env=env,
            policies={"pid": "pid", "hold": "hold"},
            seeds=(2,),
            max_steps=1,
        )
    finally:
        env.close()
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == replacement
    assert sorted(path.name for path in output.iterdir()) == [
        "comparison.json",
        "comparison.svg",
    ]


def test_compare_default_output_rejects_unmanaged_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "runs" / "three-tank" / "tracking"
    output.mkdir(parents=True)
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    env = make_env("three_tank", benchmark="tracking")
    try:
        with pytest.raises(FileExistsError, match="unmanaged entries"):
            compare_policies(
                env=env,
                policies={"pid": "pid", "hold": "hold"},
                seeds=(0,),
                max_steps=1,
            )
    finally:
        env.close()


def test_compare_requires_explicit_output_without_benchmark():
    env = make_env("quadruple")
    try:
        with pytest.raises(ValueError, match="outside a benchmark"):
            compare_policies(
                env=env,
                policies={"pid": "pid", "hold": "hold"},
                seeds=(0,),
                max_steps=1,
            )
    finally:
        env.close()

from __future__ import annotations

import copy
import importlib
import json
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from aiogym import evaluate, make_controller, make_env
from aiogym.workflows._comparison_svg import render_trajectory_svg
from aiogym.workflows.evaluate import _validation_summary
from aiogym.workflows.training_curve import _legacy_control_metrics
from aiogym.workflows.train import (
    _is_better_training_evaluation,
    _TrainingEvaluationRecorder,
)


def _plot_report(scenario, *, benchmark=None, seeds=(0,)):
    """Reuse one short hold trajectory; plot cases supply their own metrics."""
    env = make_env(scenario, benchmark=benchmark)
    try:
        template = evaluate(env=env, policies={"policy": "hold"}, seeds=(0,), max_steps=2)["evaluations"]["policy"]
    finally:
        env.close()
    report = {
        "environment": template["environment"],
        "trajectory_schema": template["trajectory_schema"],
        "seeds": list(seeds),
        "trajectory_seed": seeds[0],
        "evaluations": {},
    }
    for label in ("pid", "hold"):
        evaluation = copy.deepcopy(template)
        evaluation["policy"] = {"id": label}
        evaluation["seeds"] = list(seeds)
        evaluation["episodes"] = [
            {**copy.deepcopy(template["episodes"][0]), "seed": seed} for seed in seeds
        ]
        report["evaluations"][label] = evaluation
    return report


@pytest.mark.parametrize("change, safe, success", [
    ({}, 1, 1),
    ({"metrics": {"constraint_violations": 0, "settling_time": 90.5}}, 1, 0),
    ({"terminated": True}, 0, 0),
    ({"truncated": False}, 0, 0),
    ({"length": 199}, 0, 0),
    ({"metrics": {"constraint_violations": 1, "settling_time": 0}}, 0, 0),
    ({"length": 40, "episode_spec": {"horizon": 40},
      "metrics": {"constraint_violations": 0, "settling_time": 0}}, 1, 1),
])
def test_legacy_validation_preserves_safe_full_horizon_and_step_fraction(
    change, safe, success,
):
    episode = {
        "return": -2.0, "length": 200,
        "terminated": False, "truncated": True,
        "episode_spec": {"horizon": 200},
        "metrics": {"constraint_violations": 0, "settling_time": 90},
    }
    episode.update(change)
    episode["metrics"] = _legacy_control_metrics(episode, 0.5)
    summary = _validation_summary([episode])
    assert summary["safe_completion"] == safe
    assert summary["control_success"] == success
    assert summary["mean_return"] == -2
    assert summary["median_return"] == -2
    assert summary["p10_return"] == -2


def test_validation_summary_includes_all_cases_in_returns_and_rates():
    episodes = [{
        "return": value, "length": 200,
        "terminated": index == 3, "truncated": True,
        "episode_spec": {"horizon": 200},
        "metrics": {"constraint_violations": int(index == 3),
                    "settling_time": 0 if index < 2 else 200},
    } for index, value in enumerate([-1, -2, -4, -100])]
    for episode in episodes:
        episode["metrics"] = _legacy_control_metrics(episode, 1.0)
    summary = _validation_summary(episodes)
    assert summary["safe_completion"] == 0.75
    assert summary["control_success"] == 0.5
    assert summary["mean_return"] == -26.75
    assert summary["median_return"] == -3
    assert summary["p10_return"] == pytest.approx(-71.2)


@pytest.mark.parametrize("metrics", [
    {"safe_completion": 0}, {"control_success": 0},
    {"safe_completion": -1, "control_success": 0},
    {"safe_completion": 0, "control_success": 1},
    {"safe_completion": 1, "control_success": 0.5},
    {"safe_completion": 1, "control_success": float("nan")},
    {"safe_completion": 1, "control_success": True},
])
def test_validation_summary_requires_valid_control_metrics(metrics):
    with pytest.raises(ValueError, match="validation"):
        _validation_summary([{"return": -2, "length": 100, "metrics": metrics}])


def test_training_evaluation_prioritizes_safety():
    safe = {
        "safe_completion": 1.0,
        "control_success": 0.0,
        "mean_return": -200.0,
    }
    unsafe = {
        "safe_completion": 0.5,
        "control_success": 0.5,
        "mean_return": -50.0,
    }
    assert not _is_better_training_evaluation(unsafe, safe)
    assert _is_better_training_evaluation(safe, unsafe)


def test_training_evaluation_prioritizes_control_success_before_return():
    best = {
        "safe_completion": 1.0,
        "control_success": 0.5,
        "mean_return": -10.0,
    }
    candidate = {**best, "control_success": 0.55, "mean_return": -100.0}
    assert _is_better_training_evaluation(candidate, best)
    assert not _is_better_training_evaluation(best, candidate)


def test_training_evaluation_uses_mean_return_to_break_rate_ties():
    best = {
        "safe_completion": 0.5,
        "control_success": 0.0,
        "episode_length": 600.0,
        "mean_return": -10.0,
        "median_return": -2.0,
        "p10_return": -20.0,
    }
    candidate = {
        **best, "episode_length": 100.0, "mean_return": -9.0,
        "median_return": -5.0, "p10_return": -25.0,
    }
    assert _is_better_training_evaluation(candidate, best)
    assert not _is_better_training_evaluation(best, candidate)


def test_training_evaluation_equal_scores_keep_existing_checkpoint():
    best = {
        "safe_completion": 0.5,
        "control_success": 0.25,
        "episode_length": 100.0,
        "mean_return": -10.0,
        "median_return": -10.0,
        "p10_return": -20.0,
    }
    candidate = {**best, "episode_length": 600.0, "median_return": -5.0, "p10_return": -15.0}
    assert not _is_better_training_evaluation(candidate, best)
    assert not _is_better_training_evaluation(best, candidate)
    assert _is_better_training_evaluation(candidate, None)


def test_training_recorder_saves_only_improved_validation_candidates(tmp_path, monkeypatch):
    module = importlib.import_module("aiogym.workflows.train")
    batches = []
    for safe_count, success_count, value in (
        (10, 10, 0), (20, 0, -20), (20, 10, -30),
        (20, 10, -25), (20, 10, -26), (20, 10, -25),
    ):
        batches.append({"episodes": [{
            "seed": 1000 + index, "return": value, "length": 40,
            "terminated": index >= safe_count, "truncated": True,
            "episode_spec": {"horizon": 40}, "episode_family": "interior",
            "episode_parameters": {}, "runtime_variation": {},
            "metrics": {"safe_completion": float(index < safe_count),
                        "control_success": float(index < success_count),
                        "constraint_violations": int(index >= safe_count),
                        "settling_time": 0 if index < success_count else 40},
        } for index in range(20)]})
    evaluations = iter(batches)
    saved_steps = []
    monkeypatch.setattr(module, "_evaluate", lambda **kwargs: next(evaluations))
    monkeypatch.setattr(
        module, "save_checkpoint",
        lambda *args, **kwargs: saved_steps.append(kwargs["training"]["completed_steps"]),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(control_dt=1.0,
        reward=SimpleNamespace(success_criterion="Test criterion")))
    monkeypatch.setattr(module, "environment_metadata", lambda env: {})
    recorder = _TrainingEvaluationRecorder(
        backend=SimpleNamespace(policy=lambda *args, **kwargs: object()),
        model=object(), env=env, checkpoint_env=env, evaluate_every=1,
        seeds=range(1000, 1020), best_checkpoint=tmp_path / "best" / "model.zip",
        initial_steps=0, checkpoint_state=lambda step: {"completed_steps": step},
        checkpoint_history=lambda **kwargs: None,
    )
    recorder.start()
    for step in range(1, len(batches)):
        recorder._evaluate(step)
    history = recorder.payload()
    assert saved_steps == [0, 1, 2, 3]
    assert history["best_step"] == 3
    assert history["best_safe_completion"] == 1
    assert history["best_control_success"] == 0.5
    assert history["best_mean_return"] == -25


def test_evaluate_preserves_seed_order_and_aggregates_metrics():
    env = make_env("quadruple")
    try:
        result = evaluate(
            env=env,
            policies={"policy": "pid"},
            seeds=(4, 3),
            max_steps=4,
        )["evaluations"]["policy"]
        observation, _ = env.reset(seed=20)
        assert observation.shape == env.observation_space.shape
    finally:
        env.close()

    assert result["schema_version"] == "aiogym.evaluation.v5"
    assert result["ranking_metrics"] == [
        {"name": "return", "direction": "maximize", "aggregate": "mean"}
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
    assert result["trajectory_schema"]["time_unit"] == "s"
    assert result["trajectory_schema"]["output"][0]["low"] == 0.0
    assert result["trajectory_schema"]["output"][0]["high"] == 20.0


def test_evaluate_single_policy_keeps_full_result_with_optional_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "evaluation"
    with make_env("quadruple") as env:
        result = evaluate(env=env, policies={"hold": "hold"}, seeds=(1, 2), max_steps=2)
        assert list(tmp_path.iterdir()) == []
        exported = evaluate(env=env, policies={"hold": "hold"}, seeds=(1, 2),
                            max_steps=2, output=output)
        assert exported == result
        with pytest.raises(FileExistsError, match="non-empty directory"):
            evaluate(env=env, policies={"hold": "hold"}, seeds=(1,), output=output)
    assert result["schema_version"] == "aiogym.evaluation.v6"
    assert result["ordering"] == ["hold"]
    assert "trajectory_archive" not in result
    assert all("trajectory" in row for row in result["evaluations"]["hold"]["episodes"])
    report = json.loads((output / "comparison.json").read_text())
    assert report["schema_version"] == "aiogym.comparison.v7"
    assert len(report["trajectory_archive"]["entries"]) == 2
    assert "trajectory" not in report["evaluations"]["hold"]["episodes"][1]
    svg = (output / "comparison.svg").read_text()
    assert 'data-section="paired"' not in svg
    assert "single policy" in svg
    ET.fromstring(svg)
    assert sorted(p.name for p in output.iterdir()) == [
        "comparison.json", "comparison.svg", "trajectories.npz",
    ]


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
            evaluate(env=env, policies={"policy": "hold"}, seeds=seeds, max_steps=1)
    finally:
        env.close()


def test_evaluate_validates_the_public_policy_boundary():
    class IncompletePolicy:
        def metadata(self):
            return {"id": "incomplete"}

    class InvalidMetadataPolicy:
        env = None

        def reset(self, seed=None):
            del seed

        def act(self, observation, context):
            del context
            return np.zeros_like(observation[:2])

        def metadata(self):
            return []

    env = make_env("quadruple")
    try:
        for policies in ({}, [], "pid"):
            with pytest.raises(ValueError, match="non-empty mapping"):
                evaluate(env=env, policies=policies, seeds=[0])
        with pytest.raises(ValueError, match="policy labels"):
            evaluate(env=env, policies={"": "pid"}, seeds=[0])
        with pytest.raises(TypeError, match="policy must provide"):
            evaluate(
                env=env,
                policies={"policy": IncompletePolicy()},
                seeds=[0],
                max_steps=1,
            )
        with pytest.raises(TypeError, match="metadata.*mapping"):
            evaluate(
                env=env,
                policies={"policy": InvalidMetadataPolicy()},
                seeds=[0],
                max_steps=1,
            )
    finally:
        env.close()


def test_compare_matches_individual_evaluation_and_return_ordering(tmp_path):
    env = make_env("quadruple")
    try:
        pid = make_controller("pid", env=env)
        single = evaluate(env=env, policies={"policy": pid}, seeds=(3, 4), max_steps=3)["evaluations"]["policy"]
        comparison = evaluate(
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
    report = json.loads((tmp_path / "comparison/comparison.json").read_text())
    assert "trajectory" not in report["evaluations"]["pid"]["episodes"][1]
    archive = report["trajectory_archive"]
    archived_seed_4 = next(
        entry
        for entry in archive["entries"]
        if entry["policy"] == "pid" and entry["seed"] == 4
    )
    with np.load(tmp_path / "comparison" / archive["file"], allow_pickle=False) as data:
        prefix = archived_seed_4["id"]
        np.testing.assert_allclose(
            data[f"{prefix}__true_state"],
            single["episodes"][1]["trajectory"]["true_state"],
        )
        np.testing.assert_allclose(
            data[f"{prefix}__reward"],
            single["episodes"][1]["trajectory"]["reward"],
        )
    means = {
        label: result["aggregate"]["return"]["mean"]
        for label, result in comparison["evaluations"].items()
    }
    assert comparison["ordering"] == sorted(
        means, key=lambda label: (-means[label], label)
    )


@pytest.mark.parametrize("scenario", ("quadruple", "three_tank"))
def test_compare_reuses_identical_cases_for_every_policy(tmp_path, scenario):
    env = make_env(scenario, benchmark="tracking")
    try:
        comparison = evaluate(
            env=env,
            policies={"pid": "pid", "mpc": "mpc"},
            seeds=(0, 1, 2),
            max_steps=1,
            output=tmp_path / scenario,
        )
    finally:
        env.close()

    pid_episodes = comparison["evaluations"]["pid"]["episodes"]
    mpc_episodes = comparison["evaluations"]["mpc"]["episodes"]
    pid_specs = [episode["episode_spec"] for episode in pid_episodes]
    assert pid_specs == [episode["episode_spec"] for episode in mpc_episodes]
    assert len({repr(spec) for spec in pid_specs}) == 3
    assert [
        episode["episode_parameters"]["case_seed"]
        for episode in pid_episodes
    ] == [0, 1, 2]


def test_compare_allows_constraint_names_observed_by_only_one_policy(tmp_path):
    env = make_env("quadruple", benchmark="tracking")
    try:
        unsafe_pid = make_controller(
            "pid",
            env=env,
            config={
                "matrix_terms": [
                    {
                        "actuator": "pump_1_voltage",
                        "output": "lower_tank_1_level",
                        "kp": 0.0,
                        "ki": 0.0,
                        "kd": 0.0,
                    },
                    {
                        "actuator": "pump_2_voltage",
                        "output": "lower_tank_2_level",
                        "kp": 0.0,
                        "ki": 0.0,
                        "kd": 0.0,
                    },
                ],
                "bias": [1.0, 1.0],
            },
        )
        comparison = evaluate(
            env=env,
            policies={"unsafe": unsafe_pid, "safe": "mpc"},
            seeds=(0,),
            output=tmp_path / "comparison",
        )
    finally:
        env.close()

    assert comparison["evaluations"]["unsafe"]["episodes"][0]["terminated"]
    assert not comparison["evaluations"]["safe"]["episodes"][0]["terminated"]
    assert comparison["ordering"][0] == "safe"
    assert comparison["trajectory_schema"]["constraint_cost_names"] == [
        "tank_overflow_limit"
    ]


def test_compare_does_not_hide_one_unsafe_case_behind_a_zero_median(tmp_path):
    env = make_env("quadruple", benchmark="tracking")
    try:
        unsafe = make_controller(
            "pid",
            env=env,
            config={
                "matrix_terms": [
                    {
                        "actuator": actuator,
                        "output": output,
                        "kp": 0.0,
                        "ki": 0.0,
                        "kd": 0.0,
                    }
                    for actuator, output in (
                        ("pump_1_voltage", "lower_tank_1_level"),
                        ("pump_2_voltage", "lower_tank_2_level"),
                    )
                ],
                "bias": [1.0, 1.0],
            },
        )
        fallback = make_controller("mpc", env=env)

        class OneCaseUnsafePolicy:
            def __init__(self):
                self.env = env

            def reset(self, seed=None):
                unsafe.reset(seed)
                fallback.reset(seed)

            def act(self, observation, context):
                policy = (
                    unsafe
                    if env.unwrapped.episode_parameters["case_seed"] == 0
                    else fallback
                )
                return policy.act(observation, context)

            def metadata(self):
                return {"id": "one-case-unsafe"}

        comparison = evaluate(
            env=env,
            policies={
                "a-one-case-unsafe": OneCaseUnsafePolicy(),
                "z-all-safe": make_controller("mpc", env=env),
            },
            seeds=(0, 1, 2),
            output=tmp_path / "comparison",
        )
    finally:
        env.close()

    unsafe_rate = comparison["evaluations"]["a-one-case-unsafe"]["aggregate"][
        "unsafe_rate"
    ]
    assert unsafe_rate["median"] == 0.0
    assert unsafe_rate["mean"] > 0.0
    assert comparison["evaluations"]["z-all-safe"]["aggregate"]["unsafe_rate"][
        "mean"
    ] == 0.0
    assert comparison["ordering"][0] == "z-all-safe"


def test_quadruple_pid_safely_completes_tracking_benchmark():
    env = make_env("quadruple", benchmark="tracking")
    try:
        result = evaluate(env=env, policies={"policy": "pid"}, seeds=(0, 1, 2))["evaluations"]["policy"]
    finally:
        env.close()

    assert all(not episode["terminated"] for episode in result["episodes"])
    assert result["aggregate"]["episode_length"]["mean"] == 180.0
    assert result["aggregate"]["tracking_iae"]["mean"] < 20.0


def test_compare_uses_reward_direction_and_writes_json_and_svg(tmp_path):
    output = tmp_path / "comparison"
    env = make_env("three_tank")
    try:
        result = evaluate(
            env=env,
            policies={"pid": "pid", "hold": "hold"},
            seeds=(0, 1),
            max_steps=2,
            output=output,
        )
    finally:
        env.close()
    means = {
        label: evaluation["aggregate"]["return"]["mean"]
        for label, evaluation in result["evaluations"].items()
    }
    assert result["ranking_metrics"] == [
        {"name": "return", "direction": "maximize", "aggregate": "mean"}
    ]
    assert result["schema_version"] == "aiogym.evaluation.v6"
    report = json.loads((output / "comparison.json").read_text())
    assert report["schema_version"] == "aiogym.comparison.v7"
    assert result["trajectory_seed"] == 0
    assert report["trajectory_archive"]["schema_version"] == (
        "aiogym.trajectory-archive.v1"
    )
    assert report["trajectory_archive"]["file"] == "trajectories.npz"
    assert len(report["trajectory_archive"]["entries"]) == 4
    assert result["ordering"] == sorted(
        means, key=lambda label: (-means[label], label)
    )
    assert report["ordering"] == result["ordering"]
    svg_path = output / "comparison.svg"
    svg = svg_path.read_text(encoding="utf-8")
    root = ET.fromstring(svg)
    legend = [node for node in root.iter() if node.get("class") == "header-legend"]
    assert [node.text for node in legend] == ["pid", "hold", "reference"]
    assert [node.text for node in root.iter() if node.get("class") == "section"] == [
        "Shared-case tracking",
        "Absolute performance",
        "Paired tracking cost vs PID",
    ]
    values = {(node.get("data-policy"), node.get("data-metric")): node.get("data-value")
              for node in root.iter() if node.get("data-metric")}
    for label in ("pid", "hold"):
        assert float(values[label, "mean_return"]) == pytest.approx(means[label])
        assert float(values[label, "safe_completion"]) == 0
        assert float(values[label, "control_success"]) == 0
    row = next(node for node in root.iter() if node.get("data-valid-count") is not None)
    assert row.get("data-policy") == "hold"
    assert row.get("data-valid-count") == "0"
    assert row.get("data-excluded-count") == "2"
    assert "N/A — no comparable pairs" in " ".join(row.itertext())
    assert "trajectory seed: 0; evaluation cases: 2; baseline: pid" in svg
    assert sorted(path.name for path in output.iterdir()) == [
        "comparison.json",
        "comparison.svg",
        "trajectories.npz",
    ]

    env = make_env("three_tank")
    try:
        with pytest.raises(FileExistsError, match="non-empty directory"):
            evaluate(
                env=env,
                policies={"pid": "pid", "hold": "hold"},
                seeds=(0,),
                max_steps=1,
                output=output,
            )
    finally:
        env.close()


def test_compare_places_changing_disturbances_above_balanced_main_columns():
    report = _plot_report("three_tank", benchmark="disturbance-rejection")
    changing_disturbances = {"bv12_open": 1.0, "bv23_open": 1.0}
    for evaluation in report["evaluations"].values():
        evaluation["episodes"][0]["trajectory"]["disturbance"][-1].update(
            changing_disturbances
        )
    svg = render_trajectory_svg(report)
    ET.fromstring(svg)
    for name in changing_disturbances:
        assert f"Disturbance: {name}" in svg
        title = f">Disturbance: {name}</text>"
        panel_start = svg.index(title) + len(title)
        panel_end = svg.find('<text class="panel-title"', panel_start)
        panel = svg[panel_start:] if panel_end < 0 else svg[panel_start:panel_end]
        assert ">-0.05</text>" not in panel
        assert ">1.05</text>" not in panel
    disturbance_y = [
        float(value)
        for value in re.findall(
            r'<text class="panel-title" x="[^"]+" y="([^"]+)">Disturbance:',
            svg,
        )
    ]
    output_y = [
        float(value)
        for value in re.findall(
            r'<text class="panel-title" x="[^"]+" y="([^"]+)">Output:',
            svg,
        )
    ]
    assert disturbance_y and output_y
    assert max(disturbance_y) < min(output_y)


@pytest.mark.parametrize(
    ("scenario", "benchmark", "expected_rows"),
    (
        (
            "three_tank", "tracking",
            [
                [
                    "Output: tank_1_level [m]",
                    "Output: tank_2_level [m]",
                    "Output: tank_3_level [m]",
                ],
                [
                    "Applied action: pump_P101 [fraction]",
                    "Applied action: valve_V12 [fraction]",
                    "Applied action: valve_V23 [fraction]",
                    "Applied action: valve_V34 [fraction]",
                ],
            ],
        ),
        (
            "cascade", "disturbance-rejection",
            [
                [
                    "Disturbance: ambient_temperature",
                    "Disturbance: heater_H1_efficiency_factor",
                    "Disturbance: pump_flow_factor",
                ],
                [
                    "Output: tank_1_level [m]",
                    "Output: tank_2_level [m]",
                    "Output: tank_3_level [m]",
                ],
                [
                    "Output: tank_1_temperature [degC]",
                    "Output: tank_2_temperature [degC]",
                    "Output: tank_3_temperature [degC]",
                ],
                [
                    "Applied action: pump_P101 [fraction]",
                    "Applied action: valve_V12 [fraction]",
                    "Applied action: valve_V23 [fraction]",
                    "Applied action: valve_V34 [fraction]",
                ],
                [
                    "Applied action: heater_H1 [fraction]",
                    "Applied action: heater_H2 [fraction]",
                    "Applied action: heater_H3 [fraction]",
                ],
            ],
        ),
    ),
)
def test_compare_groups_plot_panels_by_physical_role(scenario, benchmark, expected_rows):
    report = _plot_report(scenario, benchmark=benchmark)
    if benchmark == "disturbance-rejection":
        for evaluation in report["evaluations"].values():
            evaluation["episodes"][0]["trajectory"]["disturbance"][-1].update({
                "ambient_temperature": 24.0,
                "heater_H1_efficiency_factor": 0.8,
                "pump_flow_factor": 0.9,
            })
    root = ET.fromstring(render_trajectory_svg(report))
    rows = {}
    for node in root.iter():
        if node.get("class") == "panel-title":
            rows.setdefault(float(node.get("y")), []).append(node.text)
    assert [rows[y] for y in sorted(rows)] == expected_rows


def test_extraction_plot_uses_model_time_and_control_success():
    result = _plot_report("extraction", benchmark="tracking")
    # Synthetic complete cases straddle the final 10-control-step threshold.
    for label, hold_steps in (("pid", 10), ("hold", 9)):
        case = result["evaluations"][label]["episodes"][0]
        case.update(length=100, terminated=False, truncated=True)
        case["episode_spec"]["horizon"] = 100
        case["metrics"].update(
            safe_completion=1.0, control_success=float(hold_steps == 10),
            constraint_violations=0,
            settling_time=100 * result["environment"]["control_dt"] - hold_steps * result["environment"]["control_dt"],
        )
    assert result["trajectory_schema"]["time_unit"] == "h"
    svg = render_trajectory_svg(result)
    assert ">time [h]</text>" in svg
    assert ">time [s]</text>" not in svg

    root = ET.fromstring(svg)
    assert [node.text for node in root.iter() if node.get("class") == "panel-title"] == [
        "Output: stage_5_liquid_concentration [fraction]",
        "Applied action: liquid_feed_flow [normalized_flow]",
        "Applied action: gas_feed_flow [normalized_flow]",
    ]
    assert {
        node.get("data-policy"): float(node.get("data-value"))
        for node in root.iter() if node.get("data-metric") == "control_success"
    } == {"pid": 1.0, "hold": 0.0}



def test_evaluate_uses_benchmark_ranking_without_implicit_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with make_env("quadruple", benchmark="disturbance-rejection") as env:
        result = evaluate(env=env, policies={"pid": "pid", "hold": "hold"},
                          seeds=(0, 1), max_steps=2)
    assert result["ranking_metrics"] == [
        {"name": "unsafe_rate", "direction": "minimize", "aggregate": "mean"},
        {"name": "return", "direction": "maximize", "aggregate": "mean"},
    ]
    assert "return" in result["evaluations"]["pid"]["aggregate"]
    assert all("trajectory" in episode for evaluation in result["evaluations"].values()
               for episode in evaluation["episodes"])
    assert list(tmp_path.iterdir()) == []


def test_evaluate_without_output_does_not_touch_old_benchmark_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "runs/three-tank/benchmarks/tracking"
    output.mkdir(parents=True)
    (output / "comparison.json").write_text("previous report")
    (output / "notes.txt").write_text("keep")
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    with make_env("three_tank", benchmark="tracking") as env:
        evaluate(env=env, policies={"pid": "pid"}, seeds=(0,), max_steps=1)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original


def test_compare_mean_ranking_and_paired_scores_use_matching_cases(tmp_path, monkeypatch):
    module = importlib.import_module("aiogym.workflows.evaluate")
    env = make_env("quadruple", benchmark="tracking")
    try:
        template = evaluate(env=env, policies={"policy": "pid"}, seeds=(0, 1, 2), max_steps=2)["evaluations"]["policy"]
        batches = {}
        for label, returns in (("candidate", [-4, -4, -4]), ("reference", [-1, -1, -50])):
            batch = copy.deepcopy(template)
            batch["policy"] = {"id": "pid" if label == "reference" else "custom"}
            for episode, value in zip(batch["episodes"], returns):
                episode.update({"return": value, "truncated": True})
                episode["episode_spec"]["horizon"] = 2
                episode["metrics"].update({"return": value, "tracking_ise": -2 * value, "safe_completion": 1.0})
            for key in ("return", "episode_return"):
                batch["aggregate"][key].update(mean=float(np.mean(returns)), median=float(np.median(returns)))
            batch["return_distribution"] = returns
            batches[label] = batch
        monkeypatch.setattr(module, "_evaluate", lambda **kwargs: batches[kwargs["policy"]])
        result = evaluate(
            env=env, policies={"candidate": "candidate", "reference": "reference"},
            seeds=(0, 1, 2), output=tmp_path / "paired",
        )
        assert result["ordering"] == ["candidate", "reference"]
        assert batches["candidate"]["aggregate"]["return"]["median"] < batches["reference"]["aggregate"]["return"]["median"]
        root = ET.parse(tmp_path / "paired" / "comparison.svg").getroot()
        box = next(node for node in root.iter() if node.get("data-box-policy"))
        assert box.get("data-box-policy") == "candidate"
        assert [float(box.get(key)) for key in (
            "data-q1", "data-median", "data-q3", "data-whisker-low", "data-whisker-high",
        )] == pytest.approx([2.04, 4, 4, 0.08, 4])
        assert [float(node.get("data-value")) for node in root.iter()
                if node.get("data-metric") == "mean_return"] == pytest.approx([-4, -52 / 3])
        batches["reference"]["episodes"][0]["episode_spec"]["reference"][0] += 1
        with pytest.raises(ValueError, match="identical physical cases"):
            evaluate(
                env=env, policies={"candidate": "candidate", "reference": "reference"},
                seeds=(0, 1, 2), output=tmp_path / "different-cases",
            )
    finally:
        env.close()


def test_paired_plot_marks_zero_costs_and_noncomparable_cases():
    report = _plot_report("quadruple", seeds=range(7))
    for label, evaluation in report["evaluations"].items():
        for episode in evaluation["episodes"]:
            episode["episode_spec"]["horizon"] = 2
            episode["truncated"] = True
            episode["metrics"]["safe_completion"] = 1.0
            episode["metrics"]["tracking_ise"] = 1 if label == "pid" else 0
    pid, hold = (report["evaluations"][label]["episodes"] for label in ("pid", "hold"))
    pid[0]["metrics"]["tracking_ise"] = 0
    pid[1]["metrics"]["tracking_ise"] = 1e-8
    hold[1]["metrics"]["tracking_ise"] = 1
    hold[2]["metrics"]["constraint_violations"] = 1
    hold[3]["terminated"] = True
    hold[4]["truncated"] = False
    pid[5]["terminated"] = True
    for episode in [*hold[2:5], pid[5]]:
        episode["metrics"]["safe_completion"] = 0.0
    root = ET.fromstring(render_trajectory_svg(report))
    row = next(node for node in root.iter() if node.get("data-valid-count") is not None)
    assert (row.get("data-valid-count"), row.get("data-excluded-count")) == ("2", "5")
    text = " ".join(row.itertext())
    assert "seed 0: ref=0" in text
    assert all(f"seed {seed}: unsafe or incomplete pair" in text for seed in (2, 3, 4, 5))
    box = next(node for node in root.iter() if node.get("data-box-policy"))
    assert [float(box.get(key)) for key in (
        "data-q1", "data-median", "data-q3", "data-whisker-low", "data-whisker-high",
    )] == pytest.approx([2.5e7, 5e7, 7.5e7, 0, 1e8])
    assert "linear ≤1×, log >1×" in " ".join(root.itertext())


def test_paired_boxes_use_raw_quartiles_and_adapt_to_ratio_spread():
    report = _plot_report("quadruple", seeds=range(20))
    for evaluation in report["evaluations"].values():
        for episode in evaluation["episodes"]:
            episode["episode_spec"]["horizon"] = 2
            episode["truncated"] = True
            episode["metrics"]["safe_completion"] = 1.0
            episode["metrics"]["tracking_ise"] = 1
    for values in (np.r_[np.arange(1, 20), 100], np.ones(20),
                   np.linspace(0.99990, 0.99997, 20), np.geomspace(1e-5, 1e4, 20),
                   np.r_[1e-9, np.linspace(0.98, 1.02, 18), 56.7]):
        for episode, value in zip(report["evaluations"]["hold"]["episodes"], values):
            episode["metrics"]["tracking_ise"] = float(value)
        root = ET.fromstring(render_trajectory_svg(report))
        box = next(node for node in root.iter() if node.get("data-box-policy"))
        assert [float(box.get(key)) for key in ("data-q1", "data-median", "data-q3")] == pytest.approx(
            np.quantile(values, [0.25, 0.5, 0.75]),
        )
        marks = [node for node in root.iter() if node.get("data-seed")]
        assert max(float(node.get("data-ratio")) for node in marks) == max(values)
        ticks = [node.text for node in root.iter() if node.get("class") == "legend"
                 and node.get("text-anchor") == "middle"]
        assert len(set(ticks)) == len(ticks) >= 3
        if min(values) == 1e-9:
            # Near-perfect cases must retain their original ratios.
            assert min(float(node.get("data-ratio")) for node in marks) == 1e-9
        if max(values) == 100:
            assert [float(box.get(key)) for key in (
                "data-q1", "data-median", "data-q3", "data-whisker-low", "data-whisker-high",
            )] == [5.75, 10.5, 15.25, 1, 19]
            assert [(node.get("data-seed"), node.get("data-outlier")) for node in marks] == [("19", "true")]


def test_evaluate_multiple_policies_without_benchmark_or_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with make_env("quadruple") as env:
        result = evaluate(env=env, policies={"pid": "pid", "hold": "hold"},
                          seeds=(0,), max_steps=1)
    assert set(result["evaluations"]) == {"pid", "hold"}
    assert list(tmp_path.iterdir()) == []


def test_batch_comparison_reports_endpoint_quality_and_task_success(tmp_path):
    env = make_env("crystallization", benchmark="tracking")
    try:
        result = evaluate(env=env, policies={"pid": "pid", "hold": "hold"},
                                  seeds=(0,), output=tmp_path / "batch")
        assert [row["name"] for row in result["ranking_metrics"]] == [
            "safe_completion", "control_success", "terminal_quality_cost",
        ]
        assert result["ranking_metrics"][1]["aggregate"] == "mean"
        assert "settling_fraction" not in result
        svg = (tmp_path / "batch/comparison.svg").read_text()
        root = ET.fromstring(svg)
        assert "Paired terminal quality cost vs PID" in " ".join(root.itertext())
        assert "Endpoint coefficient of variation error [dimensionless] (median)" in " ".join(root.itertext())
        assert "Endpoint mean crystal size error [um] (median)" in " ".join(root.itertext())
        assert "Relative ISE" not in svg and "last 10%" not in svg
        for evaluation in result["evaluations"].values():
            metrics = evaluation["episodes"][0]["metrics"]
            assert "settling_time" not in metrics and "tracking_ise" not in metrics
            assert metrics["safe_completion"] == 1
    finally:
        env.close()


def test_extraction_energy_integrates_native_hours():
    env = make_env("extraction")
    try:
        result = evaluate(env=env, policies={"policy": "hold"}, seeds=(0,), max_steps=2)["evaluations"]["policy"]
        case = result["episodes"][0]
        expected = sum(
            env.unwrapped.model.energy_kw(np.asarray(action)) * env.unwrapped.control_dt
            for action in case["trajectory"]["applied_action"]
        )
        assert case["metrics"]["energy"] == pytest.approx(expected)
    finally:
        env.close()


@pytest.mark.parametrize("failure", ["render", "commit", "rollback"])
def test_evaluation_export_failures_clean_partial_files_or_preserve_recovery(
    tmp_path, monkeypatch, failure,
):
    from pathlib import Path

    module = importlib.import_module("aiogym.workflows._evaluation_report")
    output = tmp_path / "evaluation"
    (tmp_path / "notes.txt").write_text("keep")
    replace = module.os.replace
    unlink = Path.unlink

    def fail_replace(source, destination):
        if destination == output / "comparison.svg":
            raise OSError("injected replacement failure")
        return replace(source, destination)

    def fail_unlink(path, *args, **kwargs):
        if path == output / "trajectories.npz":
            raise OSError("injected cleanup failure")
        return unlink(path, *args, **kwargs)

    def fail_render(result):
        raise RuntimeError("injected rendering failure")

    if failure == "render":
        monkeypatch.setattr(module, "render_trajectory_svg", fail_render)
    else:
        monkeypatch.setattr(module.os, "replace", fail_replace)
        if failure == "rollback":
            monkeypatch.setattr(Path, "unlink", fail_unlink)
    message = {"render": "injected rendering", "commit": "injected replacement",
               "rollback": "recovery files retained"}[failure]
    with make_env("heater", benchmark="tracking") as env:
        with pytest.raises((RuntimeError, OSError), match=message):
            evaluate(env=env, policies={"pid": "pid"}, seeds=(1,),
                     max_steps=2, output=output)
        assert (tmp_path / "notes.txt").read_text() == "keep"
        if failure == "rollback":
            pending = output / ".comparison-pending"
            assert (pending / "comparison.json").is_file()
            assert (pending / "comparison.svg").is_file()
            assert (output / "trajectories.npz").is_file()
            with pytest.raises(FileExistsError, match="active or unfinished"):
                evaluate(env=env, policies={"pid": "pid"}, seeds=(2,), output=output)
        else:
            assert list(output.iterdir()) == []


def test_training_validation_matches_metrics_without_building_trajectory_reports(monkeypatch):
    module = importlib.import_module("aiogym.workflows.evaluate")
    env = make_env("heater", randomize=True)
    try:
        full = evaluate(env=env, policies={"policy": "hold"}, seeds=[0, 1], max_steps=3)["evaluations"]["policy"]
        def unexpected_report(*args, **kwargs):
            raise AssertionError("training validation must not build trajectory reports")
        monkeypatch.setattr(module, "_trajectory", unexpected_report)
        monkeypatch.setattr(module, "_trajectory_summary", unexpected_report)
        compact = module._evaluate(
            env=env, policy="hold", seeds=[0, 1], max_steps=3, include_trajectories=False,
        )
        assert compact["aggregate"] == full["aggregate"]
        assert compact["episodes"] == [
            {key: value for key, value in row.items() if key != "trajectory"}
            for row in full["episodes"]
        ]
        assert _validation_summary(compact["episodes"]) == _validation_summary(full["episodes"])
    finally:
        env.close()

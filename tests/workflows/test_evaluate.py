from __future__ import annotations

import copy
import importlib
import json
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from aiogym import compare_policies, evaluate, make_controller, make_env
from aiogym.workflows._comparison_svg import render_trajectory_svg
from aiogym.workflows.evaluate import _validation_summary
from aiogym.workflows.train import (
    _is_better_training_evaluation,
    _TrainingEvaluationRecorder,
)


def _plot_report(scenario, *, benchmark=None, seeds=(0,)):
    """Reuse one short hold trajectory; plot cases supply their own metrics."""
    env = make_env(scenario, benchmark=benchmark)
    try:
        template = evaluate(env=env, policy="hold", seeds=(0,), max_steps=2)
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
def test_validation_summary_uses_safe_full_horizon_and_step_fraction(
    change, safe, success,
):
    episode = {
        "return": -2.0, "length": 200,
        "terminated": False, "truncated": True,
        "episode_spec": {"horizon": 200},
        "metrics": {"constraint_violations": 0, "settling_time": 90},
    }
    episode.update(change)
    summary = _validation_summary([episode], control_dt=0.5)
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
    summary = _validation_summary(episodes, control_dt=1.0)
    assert summary["safe_completion"] == 0.75
    assert summary["control_success"] == 0.5
    assert summary["mean_return"] == -26.75
    assert summary["median_return"] == -3
    assert summary["p10_return"] == pytest.approx(-71.2)


@pytest.mark.parametrize("metrics", [
    {"constraint_violations": 0}, {"settling_time": 0},
    {"constraint_violations": -1, "settling_time": 0},
    {"constraint_violations": 0, "settling_time": -1},
    {"constraint_violations": 0, "settling_time": 101},
    {"constraint_violations": 0, "settling_time": float("nan")},
    {"constraint_violations": 0, "settling_time": True},
])
def test_validation_summary_requires_valid_control_metrics(metrics):
    episode = {
        "return": -2, "length": 100, "terminated": False, "truncated": True,
        "episode_spec": {"horizon": 100}, "metrics": metrics,
    }
    with pytest.raises(ValueError, match="validation"):
        _validation_summary([episode], control_dt=1.0)


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
            "metrics": {"constraint_violations": int(index >= safe_count),
                        "settling_time": 0 if index < success_count else 40},
        } for index in range(20)]})
    evaluations = iter(batches)
    saved_steps = []
    monkeypatch.setattr(module, "_evaluate", lambda **kwargs: next(evaluations))
    monkeypatch.setattr(
        module, "save_checkpoint",
        lambda *args, **kwargs: saved_steps.append(kwargs["training"]["completed_steps"]),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(control_dt=1.0))
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
            policy="pid",
            seeds=(4, 3),
            max_steps=4,
        )
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
        with pytest.raises(TypeError, match="policy must provide"):
            evaluate(
                env=env,
                policy=IncompletePolicy(),
                seeds=[0],
                max_steps=1,
            )
        with pytest.raises(TypeError, match="metadata.*mapping"):
            evaluate(
                env=env,
                policy=InvalidMetadataPolicy(),
                seeds=[0],
                max_steps=1,
            )
    finally:
        env.close()


def test_compare_matches_individual_evaluation_and_return_ordering(tmp_path):
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
    compact_pid = comparison["evaluations"]["pid"]
    assert compact_pid["aggregate"] == single["aggregate"]
    assert compact_pid["policy"] == single["policy"]
    assert compact_pid["episodes"][0]["trajectory"] == single["episodes"][0][
        "trajectory"
    ]
    assert "trajectory" not in compact_pid["episodes"][1]
    assert "trajectory_summary" not in compact_pid
    archive = comparison["trajectory_archive"]
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
        comparison = compare_policies(
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
        comparison = compare_policies(
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

        comparison = compare_policies(
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
        result = evaluate(env=env, policy="pid", seeds=(0, 1, 2))
    finally:
        env.close()

    assert all(not episode["terminated"] for episode in result["episodes"])
    assert result["aggregate"]["episode_length"]["mean"] == 180.0
    assert result["aggregate"]["tracking_iae"]["mean"] < 20.0


def test_compare_uses_reward_direction_and_writes_json_and_svg(tmp_path):
    output = tmp_path / "comparison"
    env = make_env("three_tank")
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
    means = {
        label: evaluation["aggregate"]["return"]["mean"]
        for label, evaluation in result["evaluations"].items()
    }
    assert result["ranking_metrics"] == [
        {"name": "return", "direction": "maximize", "aggregate": "mean"}
    ]
    assert result["schema_version"] == "aiogym.comparison.v6"
    assert result["trajectory_seed"] == 0
    assert result["trajectory_archive"]["schema_version"] == (
        "aiogym.trajectory-archive.v1"
    )
    assert result["trajectory_archive"]["file"] == "trajectories.npz"
    assert len(result["trajectory_archive"]["entries"]) == 4
    assert result["ordering"] == sorted(
        means, key=lambda label: (-means[label], label)
    )
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == result
    svg_path = output / "comparison.svg"
    ET.parse(svg_path)
    svg = svg_path.read_text(encoding="utf-8")
    assert 'width="1528"' in svg
    root = ET.fromstring(svg)
    heading = next(node for node in root.iter() if node.get("class") == "title")
    assert heading.text == heading.text.upper()
    assert heading.get("y") == "60"
    subtitle = next(node for node in root.iter() if (node.text or "").startswith("trajectory seed:"))
    assert subtitle.get("y") == "96"
    divider = next(node for node in root.iter() if node.get("class") == "section-divider")
    assert divider.get("y1") == "128"
    assert 'font-size:34px' in svg
    legend = [node for node in root.iter() if node.get("class") == "header-legend"]
    assert [node.text for node in legend] == ["pid", "hold", "reference"]
    assert all(float(node.get("y")) > float(heading.get("y")) for node in legend)
    assert [node.text for node in root.iter() if node.get("class") == "section"] == [
        "Shared-case tracking",
        "Absolute performance",
        "Paired tracking cost vs PID",
    ]
    assert [node.text for node in root.iter() if node.get("class") == "section-number"] == [
        "01", "02", "03",
    ]
    assert len([node for node in root.iter() if node.get("class") == "section-divider"]) == 3
    bodies = {node.get("data-section"): node for node in root.iter() if node.get("data-section")}
    assert {key: node.get("transform") for key, node in bodies.items()} == {
        "tracking": "translate(56 48)", "summary": "translate(56 102)", "paired": "translate(56 155)",
    }
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
    x_axis_starts = [node.text for node in bodies["tracking"].iter()
                     if node.get("class") == "tick" and node.get("text-anchor") == "middle"
                     and node.get("x") == "76.00"]
    assert x_axis_starts
    assert set(x_axis_starts) == {"0"}
    first_level = svg.split(">Output: tank_1_level [m]</text>", 1)[1]
    first_level = first_level.split('<text class="panel-title"', 1)[0]
    level_ticks = re.findall(
        r'<text class="tick" text-anchor="end" x="68\.0" y="[^"]+">([^<]+)</text>',
        first_level,
    )
    assert level_ticks == ["0", "0.167", "0.333", "0.5"]
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
        "trajectories.npz",
    ]

    env = make_env("three_tank")
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


@pytest.mark.parametrize("benchmark", ("tracking", "boundary-safety"))
def test_three_tank_hydraulic_benchmarks_use_compact_plot_layout(
    benchmark,
):
    svg = render_trajectory_svg(_plot_report("three_tank", benchmark=benchmark))
    assert 'width="1528" height="1158"' in svg
    titles = re.findall(
        r'<text class="panel-title" x="[^"]+" y="([^"]+)">([^<]+)</text>',
        svg,
    )
    assert [title for y, title in titles if y == "148"] == [
        "Output: tank_1_level [m]",
        "Output: tank_2_level [m]",
        "Output: tank_3_level [m]",
    ]
    assert [title for y, title in titles if y == "373"] == [
        "Applied action: pump_P101 [fraction]",
        "Applied action: valve_V12 [fraction]",
        "Applied action: valve_V23 [fraction]",
        "Applied action: valve_V34 [fraction]",
    ]


@pytest.mark.parametrize(
    ("benchmark", "height", "row_titles"),
    (
        (
            "tracking",
            "1608",
            {
                "148": [
                    "Output: tank_1_level [m]",
                    "Output: tank_2_level [m]",
                    "Output: tank_3_level [m]",
                ],
                "373": [
                    "Output: tank_1_temperature [degC]",
                    "Output: tank_2_temperature [degC]",
                    "Output: tank_3_temperature [degC]",
                ],
                "598": [
                    "Applied action: pump_P101 [fraction]",
                    "Applied action: valve_V12 [fraction]",
                    "Applied action: valve_V23 [fraction]",
                    "Applied action: valve_V34 [fraction]",
                ],
                "823": [
                    "Applied action: heater_H1 [fraction]",
                    "Applied action: heater_H2 [fraction]",
                    "Applied action: heater_H3 [fraction]",
                ],
            },
        ),
        (
            "disturbance-rejection",
            "1833",
            {
                "148": [
                    "Disturbance: ambient_temperature",
                    "Disturbance: heater_H1_efficiency_factor",
                    "Disturbance: pump_flow_factor",
                ],
                "373": [
                    "Output: tank_1_level [m]",
                    "Output: tank_2_level [m]",
                    "Output: tank_3_level [m]",
                ],
                "598": [
                    "Output: tank_1_temperature [degC]",
                    "Output: tank_2_temperature [degC]",
                    "Output: tank_3_temperature [degC]",
                ],
                "823": [
                    "Applied action: pump_P101 [fraction]",
                    "Applied action: valve_V12 [fraction]",
                    "Applied action: valve_V23 [fraction]",
                    "Applied action: valve_V34 [fraction]",
                ],
                "1048": [
                    "Applied action: heater_H1 [fraction]",
                    "Applied action: heater_H2 [fraction]",
                    "Applied action: heater_H3 [fraction]",
                ],
            },
        ),
    ),
)
def test_cascade_benchmarks_group_plot_panels_by_physical_role(
    benchmark,
    height,
    row_titles,
):
    report = _plot_report("cascade", benchmark=benchmark)
    if benchmark == "disturbance-rejection":
        for evaluation in report["evaluations"].values():
            evaluation["episodes"][0]["trajectory"]["disturbance"][-1].update({
                "ambient_temperature": 24.0,
                "heater_H1_efficiency_factor": 0.8,
                "pump_flow_factor": 0.9,
            })
    svg = render_trajectory_svg(report)
    ET.fromstring(svg)
    assert f'width="1528" height="{height}"' in svg
    titles = re.findall(
        r'<text class="panel-title" x="[^"]+" y="([^"]+)">([^<]+)</text>',
        svg,
    )
    assert {
        y: [title for candidate_y, title in titles if candidate_y == y]
        for y in row_titles
    } == row_titles


def test_extraction_tracking_uses_model_time_and_balanced_plot_layout():
    result = _plot_report("extraction", benchmark="tracking")
    # Synthetic complete cases straddle the final 10-control-step threshold.
    for label, hold_steps in (("pid", 10), ("hold", 9)):
        case = result["evaluations"][label]["episodes"][0]
        case.update(length=100, terminated=False, truncated=True)
        case["episode_spec"]["horizon"] = 100
        case["metrics"].update(
            constraint_violations=0,
            settling_time=100 * result["environment"]["control_dt"] - hold_steps * result["environment"]["control_dt"],
        )
    assert result["trajectory_schema"]["time_unit"] == "h"
    svg = render_trajectory_svg(result)
    assert 'width="1528" height="1158"' in svg
    assert ">time [h]</text>" in svg
    assert ">time [s]</text>" not in svg
    titles = re.findall(
        r'<text class="panel-title" x="([^"]+)" y="([^"]+)">([^<]+)</text>',
        svg,
    )
    assert [title for _x, y, title in titles if y == "148"] == [
        "Output: stage_5_liquid_concentration [fraction]"
    ]
    assert [title for _x, y, title in titles if y == "373"] == [
        "Applied action: liquid_feed_flow [normalized_flow]",
        "Applied action: gas_feed_flow [normalized_flow]",
    ]

    root = ET.fromstring(svg)
    assert {
        node.get("data-policy"): float(node.get("data-value"))
        for node in root.iter() if node.get("data-metric") == "control_success"
    } == {"pid": 1.0, "hold": 0.0}



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
        {"name": "unsafe_rate", "direction": "minimize", "aggregate": "mean"},
        {"name": "return", "direction": "maximize", "aggregate": "mean"},
    ]
    assert "return" in result["evaluations"]["pid"]["aggregate"]
    output = (
        tmp_path
        / "runs"
        / "quadruple"
        / "benchmarks"
        / "disturbance-rejection"
    )
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
        "trajectories.npz",
    ]


def test_compare_default_output_preserves_unmanaged_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = (
        tmp_path / "runs" / "three-tank" / "benchmarks" / "tracking"
    )
    output.mkdir(parents=True)
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    env = make_env("three_tank", benchmark="tracking")
    try:
        compare_policies(
            env=env,
            policies={"pid": "pid", "hold": "hold"},
            seeds=(0,),
            max_steps=1,
        )
    finally:
        env.close()
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.svg").is_file()
    assert (output / "trajectories.npz").is_file()


def test_compare_mean_ranking_and_paired_scores_use_matching_cases(tmp_path, monkeypatch):
    module = importlib.import_module("aiogym.workflows.compare")
    env = make_env("quadruple", benchmark="tracking")
    try:
        template = evaluate(env=env, policy="pid", seeds=(0, 1, 2), max_steps=2)
        batches = {}
        for label, returns in (("candidate", [-4, -4, -4]), ("reference", [-1, -1, -50])):
            batch = copy.deepcopy(template)
            batch["policy"] = {"id": "pid" if label == "reference" else "custom"}
            for episode, value in zip(batch["episodes"], returns):
                episode.update({"return": value, "truncated": True})
                episode["episode_spec"]["horizon"] = 2
                episode["metrics"].update({"return": value, "tracking_ise": -2 * value})
            for key in ("return", "episode_return"):
                batch["aggregate"][key].update(mean=float(np.mean(returns)), median=float(np.median(returns)))
            batch["return_distribution"] = returns
            batches[label] = batch
        monkeypatch.setattr(module, "evaluate", lambda **kwargs: batches[kwargs["policy"]])
        result = compare_policies(
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
            compare_policies(
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
            episode["metrics"]["tracking_ise"] = 1 if label == "pid" else 0
    pid, hold = (report["evaluations"][label]["episodes"] for label in ("pid", "hold"))
    pid[0]["metrics"]["tracking_ise"] = 0
    pid[1]["metrics"]["tracking_ise"] = 1e-8
    hold[1]["metrics"]["tracking_ise"] = 1
    hold[2]["metrics"]["constraint_violations"] = 1
    hold[3]["terminated"] = True
    hold[4]["truncated"] = False
    pid[5]["terminated"] = True
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
    assert "linear scale (includes zero cost)" in " ".join(root.itertext())


def test_paired_boxes_use_raw_quartiles_and_adapt_to_ratio_spread():
    report = _plot_report("quadruple", seeds=range(20))
    for evaluation in report["evaluations"].values():
        for episode in evaluation["episodes"]:
            episode["episode_spec"]["horizon"] = 2
            episode["truncated"] = True
            episode["metrics"]["tracking_ise"] = 1
    for values in (np.r_[np.arange(1, 20), 100], np.ones(20),
                   np.linspace(0.99990, 0.99997, 20), np.geomspace(1e-5, 1e4, 20)):
        for episode, value in zip(report["evaluations"]["hold"]["episodes"], values):
            episode["metrics"]["tracking_ise"] = float(value)
        root = ET.fromstring(render_trajectory_svg(report))
        box = next(node for node in root.iter() if node.get("data-box-policy"))
        assert [float(box.get(key)) for key in ("data-q1", "data-median", "data-q3")] == pytest.approx(
            np.quantile(values, [0.25, 0.5, 0.75]),
        )
        marks = [node for node in root.iter() if node.get("data-seed")]
        assert max(float(node.get("data-ratio")) for node in marks) == max(values)
        assert all(190 <= float(node.get("cx")) <= 1200 for node in marks)
        ticks = [node.text for node in root.iter() if node.get("class") == "legend"
                 and node.get("text-anchor") == "middle"]
        assert len(set(ticks)) == len(ticks) >= 3
        if max(values) == 100:
            assert [float(box.get(key)) for key in (
                "data-q1", "data-median", "data-q3", "data-whisker-low", "data-whisker-high",
            )] == [5.75, 10.5, 15.25, 1, 19]
            assert [(node.get("data-seed"), node.get("data-outlier")) for node in marks] == [("19", "true")]


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


def test_extraction_energy_integrates_native_hours():
    env = make_env("extraction")
    try:
        result = evaluate(env=env, policy="hold", seeds=(0,), max_steps=2)
        case = result["episodes"][0]
        expected = sum(
            env.unwrapped.model.energy_kw(np.asarray(action)) * env.unwrapped.control_dt
            for action in case["trajectory"]["applied_action"]
        )
        assert case["metrics"]["energy"] == pytest.approx(expected)
    finally:
        env.close()


@pytest.mark.parametrize("failure, existing", [
    ("render", True), ("commit", True), ("commit", False), ("rollback", True),
])
def test_comparison_failures_preserve_results_or_recovery_copies(
    tmp_path, monkeypatch, failure, existing,
):
    from pathlib import Path

    module = importlib.import_module("aiogym.workflows.compare")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "runs/heater/benchmarks/tracking"
    output.mkdir(parents=True)
    (output / "notes.txt").write_text("keep")
    env = make_env("heater", benchmark="tracking")
    try:
        if existing:
            compare_policies(env=env, policies={"pid": "pid", "hold": "hold"},
                             seeds=(0,), max_steps=1)
        original = {p.name: p.read_bytes() for p in output.iterdir()}
        replacements = 0
        replace = module.os.replace

        def fail_replace(source, destination):
            nonlocal replacements
            if Path(destination).resolve().parent == output:
                replacements += 1
                if replacements == 2 or (failure == "rollback" and replacements == 3):
                    raise OSError("injected replacement failure")
            return replace(source, destination)

        def fail_render(result):
            raise RuntimeError("injected rendering failure")

        if failure == "render":
            monkeypatch.setattr(module, "render_trajectory_svg", fail_render)
        else:
            monkeypatch.setattr(module.os, "replace", fail_replace)
        message = {"render": "injected rendering", "commit": "injected replacement",
                   "rollback": "recovery files retained"}[failure]
        with pytest.raises((RuntimeError, OSError), match=message):
            compare_policies(env=env, policies={"pid": "pid", "hold": "hold"},
                             seeds=(1,), max_steps=2)
        assert (output / "notes.txt").read_text() == "keep"
        if failure == "rollback":
            pending = output / ".comparison-pending"
            for name in ("comparison.json", "comparison.svg", "trajectories.npz"):
                assert (pending / "previous" / name).read_bytes() == original[name]
            with pytest.raises(FileExistsError, match="active or unfinished"):
                compare_policies(env=env, policies={"pid": "pid", "hold": "hold"},
                                 seeds=(2,), max_steps=1)
            assert replacements == 3
        else:
            assert {p.name: p.read_bytes() for p in output.iterdir()} == original
    finally:
        env.close()


def test_training_validation_matches_metrics_without_building_trajectory_reports(monkeypatch):
    module = importlib.import_module("aiogym.workflows.evaluate")
    env = make_env("heater", randomize=True)
    try:
        full = evaluate(env=env, policy="hold", seeds=[0, 1], max_steps=3)
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
        assert _validation_summary(compact["episodes"], control_dt=env.unwrapped.control_dt) == _validation_summary(full["episodes"], control_dt=env.unwrapped.control_dt)
    finally:
        env.close()

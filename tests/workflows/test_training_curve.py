from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from aiogym import plot_training_curve


@pytest.fixture
def curve():
    return {
        "schema_version": "aiogym.training_curve.v2",
        "record_every": 20_000,
        "initial_steps": 0,
        "actual_steps": 20_000,
        "records": [{
            "start_step": 0, "end_step": 20_000, "transition_count": 20_000,
            "mean_reward": -999, "reward_std": 0,
            "minimum_reward": -999, "maximum_reward": -999,
            "completed_episodes": 0, "terminated_episodes": 0,
            "truncated_episodes": 0,
        }],
        "episodes": [],
    }


@pytest.fixture
def history():
    seeds = [1000, 1001, 1002, 1003]
    records = []
    for step, returns in ((0, [-100, -200, -300, -400]),
                          (10_000, [-2, -4, -8, -16]),
                          (20_000, [-1, -2, -6, -9])):
        episodes = [{
            "seed": seed, "return": value, "length": 225,
            "terminated": False, "truncated": True,
            "episode_spec": {"horizon": 225, "reference": [seed / 1000]},
            "runtime_variation": {},
            "metrics": {"settling_time": 225, "constraint_violations": 0},
        } for seed, value in zip(seeds, returns)]
        records.append({"step": step, "episodes": episodes})
    last = records[-1]["episodes"]
    last[0]["metrics"]["settling_time"] = 202
    last[1]["metrics"]["settling_time"] = 203
    last[2].update(length=100, terminated=True, truncated=False)
    last[2]["metrics"].update(settling_time=20, constraint_violations=1)
    last[3]["terminated"] = True
    last[3]["metrics"].update(settling_time=10, constraint_violations=1)
    return {"seeds": seeds, "records": records, "best_step": 10_000}


def _series(root, key):
    return [float(node.attrib["data-value"]) for node in root.iter()
            if node.tag.endswith("circle") and node.get("data-series") == key]


def _plot(tmp_path, curve, history, **kwargs):
    path = plot_training_curve(
        curve, evaluation_history=history, control_dt=kwargs.pop("control_dt", 1.0),
        output=tmp_path / "curve.svg", **kwargs,
    )
    return ET.parse(path).getroot()


def test_fixed_validation_mean_p10_and_success_use_all_cases(tmp_path, curve, history):
    root = _plot(tmp_path, curve, history)
    assert _series(root, "mean_return") == [-250, -7.5, -4.5]
    assert _series(root, "p10_return") == pytest.approx([
        np.quantile([e["return"] for e in row["episodes"]], 0.1)
        for row in history["records"]
    ])
    assert _series(root, "safe_completion") == [100, 100, 50]
    assert _series(root, "control_success") == [0, 0, 25]
    text = " ".join(root.itertext())
    assert "last 10% of planned control steps continuously within scenario settling tolerances" in text
    assert "selected best: 10,000 steps" in text
    assert "original selection; not reselected" in text
    assert "one training run" in text
    assert "not a confidence interval" in text
    assert "symmetric log, linear within [-1, 1]" in text
    assert "-999" not in text


def test_settling_fraction_is_independent_of_time_unit(tmp_path, curve, history):
    # 225 steps => ceil(22.5) = 23 final steps, deadline after step 202.
    for record in history["records"]:
        for episode in record["episodes"]:
            episode["metrics"]["settling_time"] *= 0.1
    root = _plot(tmp_path, curve, history, control_dt=0.1)
    assert _series(root, "control_success")[-1] == 25


def test_current_history_uses_task_metrics_without_settling_time(tmp_path, curve, history):
    history.update(schema_version="aiogym.training_evaluation.v5",
                   success_criterion="Safe full batch; endpoint quality within tolerance.")
    for record in history["records"]:
        for episode in record["episodes"]:
            episode.update(terminated=True, truncated=False, length=225)
            episode["episode_spec"]["terminal_at_horizon"] = True
            episode["metrics"] = {"safe_completion": 1.0, "control_success": 1.0}
    root = _plot(tmp_path, curve, history, control_dt=None)
    assert _series(root, "control_success") == [100, 100, 100]
    text = " ".join(root.itertext())
    assert "endpoint quality within tolerance" in text
    assert "last 10%" not in text
    with pytest.raises(ValueError, match="only to legacy"):
        plot_training_curve(curve, evaluation_history=history, settling_fraction=0.2,
                            output=tmp_path / "changed.svg")


def test_fraction_cannot_exceed_entire_episode(tmp_path, curve, history):
    with pytest.raises(ValueError, match="at most 1"):
        _plot(tmp_path, curve, history, settling_fraction=1.1)


def test_json_inputs_and_continuation_axis(tmp_path, curve, history):
    curve["initial_steps"] = 10_000
    curve["records"][0].update(start_step=10_000, transition_count=10_000)
    history["records"] = history["records"][1:]
    curve_path, history_path = tmp_path / "curve.json", tmp_path / "history.json"
    curve_path.write_text(json.dumps(curve))
    history_path.write_text(json.dumps(history))
    root = _plot(tmp_path, curve_path, history_path)
    first = next(node for node in root.iter()
                 if node.tag.endswith("circle") and node.get("data-series") == "mean_return")
    assert first.get("cx") == "92.00"
    assert _series(root, "mean_return") == [-7.5, -4.5]


def test_without_validation_does_not_plot_training_rewards(tmp_path, curve):
    path = plot_training_curve(curve, output=tmp_path / "empty.svg")
    root = ET.parse(path).getroot()
    assert "No validation data" in " ".join(root.itertext())
    assert not _series(root, "mean_return")
    assert "-999" not in " ".join(root.itertext())


@pytest.mark.parametrize("returns", [[0, 0, 0, 0], [1e-5, -1e-5, 0, 0], [1, 10, 100, 1000]])
def test_return_axis_handles_zero_small_and_positive_values(tmp_path, curve, history, returns):
    for record in history["records"]:
        for episode, value in zip(record["episodes"], returns):
            episode["return"] = value
    root = _plot(tmp_path, curve, history)
    assert _series(root, "mean_return") == [np.mean(returns)] * 3
    for node in root.iter():
        if node.tag.endswith("circle"):
            assert np.isfinite(float(node.attrib["cy"]))


@pytest.mark.parametrize("change, message", [
    (lambda h: h["records"][1]["episodes"].pop(), "same ordered seeds"),
    (lambda h: h["records"][1]["episodes"][0]["episode_spec"].update(reference=[99]), "remain fixed"),
    (lambda h: h["records"][1]["episodes"][0].update(runtime_variation={"noise": 1}), "remain fixed"),
    (lambda h: h["records"][1].update(step=0), "steps must increase"),
    (lambda h: h.update(best_step=123), "best_step"),
    (lambda h: h["records"].pop(), "initial and final"),
    (lambda h: h["records"][0]["episodes"][0]["metrics"].pop("settling_time"), "settling_time"),
    (lambda h: h["records"][0]["episodes"][0].update(length=226), "exceeds"),
])
def test_invalid_validation_inputs_do_not_create_artifacts(tmp_path, curve, history, change, message):
    history = copy.deepcopy(history)
    change(history)
    with pytest.raises((ValueError, TypeError), match=message):
        _plot(tmp_path, curve, history)
    assert not (tmp_path / "curve.svg").exists()


@pytest.mark.parametrize("kwargs", [
    {"control_dt": None}, {"control_dt": 0}, {"control_dt": float("nan")},
    {"settling_fraction": 0}, {"settling_fraction": True},
])
def test_invalid_physical_time_inputs(tmp_path, curve, history, kwargs):
    with pytest.raises((ValueError, TypeError), match="positive finite"):
        _plot(tmp_path, curve, history, **kwargs)


def test_replot_refuses_to_overwrite_existing_output(tmp_path, curve, history):
    _plot(tmp_path, curve, history)
    with pytest.raises(FileExistsError):
        _plot(tmp_path, curve, history)

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
        {"name": "return", "direction": "maximize"}
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
    medians = {
        label: result["aggregate"]["return"]["median"]
        for label, result in comparison["evaluations"].items()
    }
    assert comparison["ordering"] == sorted(
        medians, key=lambda label: (-medians[label], label)
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
    medians = {
        label: evaluation["aggregate"]["return"]["median"]
        for label, evaluation in result["evaluations"].items()
    }
    assert result["ranking_metrics"] == [
        {"name": "return", "direction": "maximize"}
    ]
    assert result["schema_version"] == "aiogym.comparison.v4"
    assert result["trajectory_seed"] == 0
    assert result["trajectory_archive"]["schema_version"] == (
        "aiogym.trajectory-archive.v1"
    )
    assert result["trajectory_archive"]["file"] == "trajectories.npz"
    assert len(result["trajectory_archive"]["entries"]) == 4
    assert result["ordering"] == sorted(
        medians, key=lambda label: (-medians[label], label)
    )
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == result
    svg_path = output / "comparison.svg"
    ET.parse(svg_path)
    svg = svg_path.read_text(encoding="utf-8")
    assert 'width="1440"' in svg
    assert (
        '<text class="panel-title" x="796.0" y="120">'
        "Applied action: pump_P101 [fraction]"
    ) in svg
    assert "Closest level-boundary distance [m]" in svg
    assert re.search(r">pid: tank [123] (?:lower|upper), ", svg)
    assert re.search(r">hold: tank [123] (?:lower|upper), ", svg)
    return_svg = svg.split(">Cumulative return by policy</text>", 1)[1]
    assert ">pid</text>" in return_svg
    assert ">hold</text>" in return_svg
    assert ">episode return (higher is better)</text>" in return_svg
    assert "circles: all return seeds; diamond: median" in return_svg
    assert "<title>seed 0:" in return_svg
    assert "<title>seed 1:" in return_svg
    series_svg = svg.split(">Cumulative return by policy</text>", 1)[0]
    assert "trajectory seed: 0; return seeds: 0, 1" in series_svg
    assert 'fill-opacity="0.16"' not in series_svg
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
    assert level_ticks == ["0", "0.167", "0.333", "0.5"]
    first_action = svg.split(">Applied action: pump_P101 [fraction]</text>", 1)[1]
    first_action = first_action.split('<text class="panel-title"', 1)[0]
    action_ticks = re.findall(
        r'<text class="tick" text-anchor="end" x="788\.0" y="[^"]+">([^<]+)</text>',
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


def test_compare_places_changing_disturbances_above_balanced_main_columns(tmp_path):
    output = tmp_path / "disturbance-comparison"
    env = make_env("three_tank", benchmark="disturbance-rejection")
    try:
        disturbance_start = min(
            env.unwrapped.default_episode.disturbance_schedule
        )
        compare_policies(
            env=env,
            policies={"pid": "pid", "hold": "hold"},
            seeds=(0,),
            max_steps=disturbance_start + 1,
            output=output,
        )
    finally:
        env.close()

    svg = (output / "comparison.svg").read_text(encoding="utf-8")
    ET.parse(output / "comparison.svg")
    assert (
        '<text class="panel-title" x="76.0" y="120">'
        "Disturbance: pump_flow_factor"
    ) in svg
    assert (
        '<text class="panel-title" x="762.0" y="120">'
        "Disturbance: v23_flow_factor"
    ) in svg
    assert (
        '<text class="panel-title" x="76.0" y="345">'
        "Output: tank_1_level [m]"
    ) in svg
    assert (
        '<text class="panel-title" x="76.0" y="570">'
        "Applied action: pump_P101 [fraction]"
    ) in svg


@pytest.mark.parametrize("benchmark", ("tracking", "boundary-safety"))
def test_three_tank_hydraulic_benchmarks_use_compact_plot_layout(
    tmp_path,
    benchmark,
):
    output = tmp_path / benchmark
    env = make_env("three_tank", benchmark=benchmark)
    try:
        compare_policies(
            env=env,
            policies={"pid": "pid", "hold": "hold"},
            seeds=(0,),
            max_steps=2,
            output=output,
        )
    finally:
        env.close()
    svg = (output / "comparison.svg").read_text(encoding="utf-8")
    assert 'width="1440" height="1060"' in svg
    titles = re.findall(
        r'<text class="panel-title" x="[^"]+" y="([^"]+)">([^<]+)</text>',
        svg,
    )
    assert [title for y, title in titles if y == "120"] == [
        "Output: tank_1_level [m]",
        "Output: tank_2_level [m]",
        "Output: tank_3_level [m]",
    ]
    assert [title for y, title in titles if y == "345"] == [
        "Applied action: pump_P101 [fraction]",
        "Applied action: valve_V12 [fraction]",
        "Applied action: valve_V23 [fraction]",
        "Applied action: valve_V34 [fraction]",
    ]
    assert [title for y, title in titles if y == "570"] == [
        "Closest level-boundary distance [m]"
    ]
    assert [title for y, title in titles if y == "797"] == [
        "Cumulative return by policy"
    ]


def test_extraction_tracking_uses_model_time_and_balanced_plot_layout(tmp_path):
    output = tmp_path / "extraction-tracking"
    env = make_env("extraction", benchmark="tracking")
    try:
        result = compare_policies(
            env=env,
            policies={"pid": "pid", "mpc": "mpc"},
            seeds=(0,),
            max_steps=2,
            output=output,
        )
    finally:
        env.close()

    assert result["trajectory_schema"]["time_unit"] == "h"
    svg = (output / "comparison.svg").read_text(encoding="utf-8")
    ET.parse(output / "comparison.svg")
    assert 'width="1440" height="1060"' in svg
    assert ">time [h]</text>" in svg
    assert ">time [s]</text>" not in svg
    titles = re.findall(
        r'<text class="panel-title" x="([^"]+)" y="([^"]+)">([^<]+)</text>',
        svg,
    )
    assert [title for _x, y, title in titles if y == "120"] == [
        "Output: stage_5_liquid_concentration [fraction]"
    ]
    assert [title for _x, y, title in titles if y == "345"] == [
        "Applied action: liquid_feed_flow [normalized_flow]",
        "Applied action: gas_feed_flow [normalized_flow]",
    ]
    assert [title for _x, y, title in titles if y == "570"] == [
        "Closest state-boundary distance [fraction]"
    ]
    assert [title for _x, y, title in titles if y == "797"] == [
        "Cumulative return by policy"
    ]


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
        {"name": "return", "direction": "maximize"},
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
    svg = (output / "comparison.svg").read_text(encoding="utf-8")
    assert "Closest level-boundary distance [cm]" in svg
    assert re.search(r">pid: tank [1234] (?:lower|upper), ", svg)

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

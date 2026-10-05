from __future__ import annotations

import importlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

pytestmark = pytest.mark.rl

import aiogym
from aiogym import load_policy, make_env, plot_training_curve, train
from aiogym.rl.algorithms import get_algorithm
from aiogym.workflows._checkpoint import load_training_checkpoint


SMALL_POLICY = {"policy_kwargs": {"net_arch": [8, 8]}}
THREE_TANK_SAC_CONFIG = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "aiogym"
        / "rl"
        / "configs"
        / "three-tank-sac-nstep10.json"
    ).read_text(encoding="utf-8")
)


def test_off_policy_training_saves_loads_and_predicts(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "sac"
    try:
        result = train(
            env=env,
            evaluate_every=None,
            algorithm="sac",
            steps=2,
            seed=4,
            algorithm_kwargs=SMALL_POLICY,
            output=output,
        )
        policy = load_policy(
            output / "model.zip",
            env=env,
        )
        observation, _ = env.reset(seed=11)
        action = policy.act(observation, {})
    finally:
        env.close()

    assert result["schema_version"] == "aiogym.training.v15"
    assert result["checkpoint_schema"] == "aiogym.checkpoint.v3"
    assert result["algorithm"] == "sac"
    assert result["steps"] == 2
    assert result["initial_steps"] == 0
    assert result["added_steps"] == 2
    assert result["actual_steps"] == 2
    assert result["resume_from"] is None
    assert result["seed"] == 4
    assert result["record_every"] == 500
    assert result["evaluation"] is None
    assert result["behavior_cloning"] is None
    assert result["behavior_cloning_artifact"] is None
    assert result["best_checkpoint"] is None
    assert result["best_tracking_figure"] is None
    assert result["evaluation_history"] is None
    assert set(path.name for path in output.iterdir()) == {
        "metadata.json",
        "model.zip",
        "training_curve.json",
        "training_curve.svg",
    }
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["algorithm"] == "sac"
    assert metadata["steps"] == 2
    assert metadata["actual_steps"] == 2
    assert metadata["seed"] == 4
    assert metadata["algorithm_kwargs"] == get_algorithm("sac").effective_kwargs(
        steps=2, values=SMALL_POLICY
    )
    assert metadata["environment"]["observation_shape"] == [8]
    assert metadata["environment"]["action_shape"] == [2]
    curve = json.loads(
        (output / "training_curve.json").read_text(encoding="utf-8")
    )
    assert curve["schema_version"] == "aiogym.training_curve.v2"
    assert curve["initial_steps"] == 0
    assert [row["end_step"] for row in curve["records"]] == [2]
    assert [row["transition_count"] for row in curve["records"]] == [2]
    with zipfile.ZipFile(output / "model.zip") as checkpoint:
        assert set(checkpoint.namelist()) == {"manifest.json", "payload.zip"}
        manifest = json.loads(checkpoint.read("manifest.json"))
    assert manifest["schema_version"] == "aiogym.checkpoint.v3"
    assert manifest["algorithm"] == "sac"
    assert manifest["environment"] == metadata["environment"]
    assert manifest["runtime"]["model_class"] == "stable_baselines3.sac.sac:SAC"
    assert manifest["policy"]["training"] == {
        "completed_steps": 2,
        "seed": 4,
        "algorithm_kwargs": result["algorithm_kwargs"],
        "dataset": None,
    }
    ET.parse(output / "training_curve.svg")


def test_sac_training_continues_optimizer_and_replay_state(tmp_path):
    env = make_env("quadruple")
    config = {
        **SMALL_POLICY,
        "batch_size": 2,
        "buffer_size": 32,
        "learning_starts": 0,
    }
    first_output = tmp_path / "first"
    continued_output = tmp_path / "continued"
    try:
        first = train(
            env=env,
            evaluate_every=None,
            algorithm="sac",
            steps=3,
            seed=6,
            algorithm_kwargs=config,
            record_every=2,
            output=first_output,
        )
        _, _, first_model, first_state = load_training_checkpoint(
            first["checkpoint"], env=env, algorithm="sac"
        )
        first_parameters = [
            value.detach().cpu().clone()
            for value in first_model.actor.parameters()
        ]

        continued = train(
            env=env,
            evaluate_every=None,
            algorithm="sac",
            steps=2,
            resume_from=first["checkpoint"],
            record_every=1,
            output=continued_output,
        )
        _, _, continued_model, continued_state = load_training_checkpoint(
            continued["checkpoint"], env=env, algorithm="sac"
        )
        third = train(
            env=env, evaluate_every=None, algorithm="sac", steps=1,
            resume_from=continued["checkpoint"], record_every=1,
            output=tmp_path / "third",
        )
    finally:
        env.close()

    curve = json.loads(
        (continued_output / "training_curve.json").read_text(encoding="utf-8")
    )
    assert first_state["completed_steps"] == 3
    assert continued["initial_steps"] == 3
    assert continued["added_steps"] == 2
    assert continued["actual_steps"] == 5
    assert continued["resume_from"] == str(
        (first_output / "model.zip").resolve()
    )
    assert continued["algorithm_kwargs"] == first["algorithm_kwargs"]
    assert continued_state["completed_steps"] == 5
    assert continued_model.num_timesteps == 5
    assert continued_model.replay_buffer.size() == 5
    assert curve["initial_steps"] == 0
    assert [row["start_step"] for row in curve["records"]] == [0, 2, 3, 4]
    assert [row["end_step"] for row in curve["records"]] == [2, 3, 4, 5]
    third_curve = json.loads(Path(third["training_curve"]).read_text())
    assert third_curve["records"][:-1] == curve["records"]
    assert third_curve["actual_steps"] == 6
    assert any(
        not np.array_equal(before.numpy(), after.detach().cpu().numpy())
        for before, after in zip(first_parameters, continued_model.actor.parameters())
    )


def test_interrupted_training_preserves_checkpoint_history_when_moved(tmp_path, monkeypatch):
    workflow = importlib.import_module("aiogym.workflows.train")
    monkeypatch.setattr(workflow, "_evaluate", partial(workflow._evaluate, max_steps=2))
    monkeypatch.setattr(
        workflow, "_is_better_training_evaluation",
        lambda candidate, best: best is None or candidate["step"] == 3,
    )
    backend_type = type(get_algorithm("sac"))
    learn = backend_type.learn

    def interrupted_learn(self, model, *, steps, dataset, on_step):
        def callback(event):
            on_step(event)
            if event.step == 4:
                raise RuntimeError("simulated interruption")
        return learn(self, model, steps=steps, dataset=dataset, on_step=callback)

    env = make_env("quadruple")
    evaluation_env = make_env("quadruple", randomize=True)
    source = tmp_path / "interrupted"
    try:
        with monkeypatch.context() as interruption:
            interruption.setattr(backend_type, "learn", interrupted_learn)
            with pytest.raises(RuntimeError, match="simulated interruption"):
                train(
                    env=env, algorithm="sac", steps=6, seed=0,
                    algorithm_kwargs={**SMALL_POLICY, "buffer_size": 32},
                    evaluation_env=evaluation_env, evaluate_every=3,
                    record_every=2, output=source,
                )
        saved_curve = (source / "training_curve.json").read_bytes()
        assert json.loads(saved_curve)["actual_steps"] == 4
        saved_history = json.loads((source / "evaluation_history.json").read_text())
        assert [row["step"] for row in saved_history["records"]] == [0, 3]
        portable = tmp_path / "portable/model.zip"
        portable.parent.mkdir()
        shutil.copyfile(source / "best/model.zip", portable)
        continued = train(
            env=env, algorithm="sac", steps=2, resume_from=portable,
            evaluation_env=evaluation_env, evaluate_every=3, record_every=2,
            output=tmp_path / "continued",
        )
    finally:
        env.close()
        evaluation_env.close()
    curve = json.loads(Path(continued["training_curve"]).read_text())
    assert curve["initial_steps"] == 0
    assert [(r["start_step"], r["end_step"]) for r in curve["records"]] == [(0, 2), (2, 3), (3, 5)]
    history = json.loads(Path(continued["evaluation_history"]).read_text())
    assert [r["step"] for r in history["records"]] == [0, 3, 5]
    assert history["records"][:2] == saved_history["records"]
    assert (source / "training_curve.json").read_bytes() == saved_curve
    ET.parse(continued["training_curve_figure"])


def test_final_checkpoint_continuation_keeps_historical_best(tmp_path, monkeypatch):
    workflow = importlib.import_module("aiogym.workflows.train")
    monkeypatch.setattr(workflow, "_evaluate", partial(workflow._evaluate, max_steps=2))
    monkeypatch.setattr(
        workflow, "_is_better_training_evaluation",
        lambda candidate, best: best is None or candidate["step"] == 1,
    )
    env = make_env("quadruple")
    evaluation_env = make_env("quadruple", randomize=True)
    try:
        first = train(
            env=env, algorithm="sac", steps=3,
            algorithm_kwargs={**SMALL_POLICY, "buffer_size": 32},
            evaluation_env=evaluation_env, evaluate_every=1,
            record_every=2, output=tmp_path / "first",
        )
        continued = train(
            env=env, algorithm="sac", steps=2, resume_from=first["checkpoint"],
            evaluation_env=evaluation_env, evaluate_every=1,
            record_every=2, output=tmp_path / "continued",
        )
    finally:
        env.close()
        evaluation_env.close()
    first_history = json.loads(Path(first["evaluation_history"]).read_text())
    history = json.loads(Path(continued["evaluation_history"]).read_text())
    assert [r["step"] for r in history["records"]] == [0, 1, 2, 3, 4, 5]
    assert history["records"][:4] == first_history["records"]
    assert continued["evaluation"]["best_step"] == 1
    assert Path(continued["best_checkpoint"]).read_bytes() == Path(first["best_checkpoint"]).read_bytes()


def test_legacy_checkpoint_uses_sibling_history_and_rejects_missing_records(tmp_path):
    env = make_env("quadruple")
    try:
        first = train(
            env=env, algorithm="sac", steps=3, evaluate_every=None,
            algorithm_kwargs={**SMALL_POLICY, "buffer_size": 32},
            record_every=2, output=tmp_path / "first",
        )
        checkpoint = Path(first["checkpoint"])
        with zipfile.ZipFile(checkpoint) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            payload = archive.read("payload.zip")
        del manifest["policy"]["training_history"]
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("payload.zip", payload)
        continued = train(
            env=env, algorithm="sac", steps=1, evaluate_every=None,
            resume_from=checkpoint, output=tmp_path / "continued",
        )
        standalone = tmp_path / "standalone/model.zip"
        standalone.parent.mkdir()
        shutil.copyfile(checkpoint, standalone)
        with pytest.raises(ValueError, match="no saved training history"):
            train(
                env=env, algorithm="sac", steps=1, evaluate_every=None,
                resume_from=standalone, output=tmp_path / "missing-history",
            )
    finally:
        env.close()
    curve = json.loads(Path(continued["training_curve"]).read_text())
    first_curve = json.loads(Path(first["training_curve"]).read_text())
    assert curve["records"][:2] == first_curve["records"]
    assert curve["initial_steps"] == 0 and curve["actual_steps"] == 4
    assert not (tmp_path / "missing-history").exists()


def test_continued_training_rejects_changed_seed_or_configuration(tmp_path):
    env = make_env("quadruple")
    randomized_env = make_env("quadruple", randomize=True)
    boundary_env = make_env(
        "quadruple", randomize=True, boundary_probability=0.3
    )
    try:
        result = train(
            env=env,
            evaluate_every=None,
            algorithm="sac",
            steps=1,
            seed=3,
            algorithm_kwargs=SMALL_POLICY,
            output=tmp_path / "first",
        )
        with pytest.raises(ValueError, match="seed must match"):
            train(
                env=env,
                evaluate_every=None,
                algorithm="sac",
                steps=1,
                seed=4,
                resume_from=result["checkpoint"],
                output=tmp_path / "seed-mismatch",
            )
        with pytest.raises(ValueError, match="algorithm_kwargs must match"):
            train(
                env=env,
                evaluate_every=None,
                algorithm="sac",
                steps=1,
                algorithm_kwargs={"policy_kwargs": {"net_arch": [4, 4]}},
                resume_from=result["checkpoint"],
                output=tmp_path / "config-mismatch",
            )
        with pytest.raises(ValueError, match="checkpoint algorithm"):
            train(
                env=env,
                evaluate_every=None,
                algorithm="td3",
                steps=1,
                resume_from=result["checkpoint"],
                output=tmp_path / "algorithm-mismatch",
            )
        with pytest.raises(ValueError, match="checkpoint randomize"):
            train(
                env=randomized_env,
                evaluate_every=None,
                algorithm="sac",
                steps=1,
                resume_from=result["checkpoint"],
                output=tmp_path / "environment-mismatch",
            )
        randomized_result = train(
            env=randomized_env,
            evaluate_every=None,
            algorithm="sac",
            steps=1,
            seed=3,
            algorithm_kwargs=SMALL_POLICY,
            output=tmp_path / "randomized",
        )
        with pytest.raises(ValueError, match="checkpoint boundary_probability"):
            train(
                env=boundary_env,
                evaluate_every=None,
                algorithm="sac",
                steps=1,
                resume_from=randomized_result["checkpoint"],
                output=tmp_path / "boundary-mismatch",
            )
    finally:
        boundary_env.close()
        randomized_env.close()
        env.close()


def test_checkpoint_rejects_incompatible_model_parameters(tmp_path):
    training_env = make_env("quadruple")
    incompatible_env = make_env(
        "quadruple",
        parameters={"pump_gain": [3.2, 3.2]},
    )
    benchmark_env = make_env("quadruple", benchmark="tracking")
    try:
        result = train(
            env=training_env,
            evaluate_every=None,
            algorithm="sac",
            steps=2,
            algorithm_kwargs=SMALL_POLICY,
            output=tmp_path / "training",
        )
        with pytest.raises(ValueError, match="checkpoint parameters"):
            load_policy(result["checkpoint"], env=incompatible_env)
        assert isinstance(load_policy(result["checkpoint"], env=benchmark_env), aiogym.Policy)
    finally:
        benchmark_env.close()
        incompatible_env.close()
        training_env.close()


def test_training_periodically_evaluates_and_saves_best_checkpoint(tmp_path, monkeypatch):
    # Keep all validation seeds, but exercise workflow wiring with short rollouts.
    module = importlib.import_module("aiogym.workflows.train")
    monkeypatch.setattr(module, "_evaluate", partial(module._evaluate, max_steps=2))
    env = make_env("quadruple")
    evaluation_env = make_env("quadruple", randomize=True)
    output = tmp_path / "evaluated-sac"
    try:
        result = train(
            env=env,
            algorithm="sac",
            steps=2,
            seed=0,
            algorithm_kwargs=SMALL_POLICY,
            evaluation_env=evaluation_env,
            evaluate_every=1,
            output=output,
        )
    finally:
        evaluation_env.close()
        env.close()

    history = json.loads(
        (output / "evaluation_history.json").read_text(encoding="utf-8")
    )
    assert history["schema_version"] == "aiogym.training_evaluation.v5"
    assert [row["step"] for row in history["records"]] == [0, 1, 2]
    assert history["seeds"] == list(range(1_000, 1_020))
    assert all(len(row["episodes"]) == 20 for row in history["records"])
    assert all(
        [episode["seed"] for episode in row["episodes"]]
        == list(range(1_000, 1_020))
        for row in history["records"]
    )
    first_cases = [
        episode["episode_spec"] for episode in history["records"][0]["episodes"]
    ]
    assert first_cases[0] != first_cases[1]
    assert all(
        [episode["episode_spec"] for episode in row["episodes"]] == first_cases
        for row in history["records"][1:]
    )
    assert history["ranking_metric"] == {
        "name": "return",
        "aggregate": "mean",
        "direction": "maximize",
    }
    assert "last 10%" in history["success_criterion"]
    assert history["selection_order"] == [
        {"name": "safe_completion", "direction": "maximize"},
        {"name": "control_success", "direction": "maximize"},
        {"name": "mean_return", "direction": "maximize"},
    ]
    assert result["evaluation"]["selection_order"] == history["selection_order"]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["evaluation"]["selection_order"] == history["selection_order"]
    for record in history["records"]:
        assert record["episode_length"] == pytest.approx(
            sum(episode["length"] for episode in record["episodes"]) / 20
        )
        returns = [episode["return"] for episode in record["episodes"]]
        assert record["mean_return"] == pytest.approx(np.mean(returns))
        assert record["median_return"] == pytest.approx(np.median(returns))
        assert record["p10_return"] == pytest.approx(np.quantile(returns, 0.1))
        assert 0 <= record["control_success"] <= record["safe_completion"] <= 1
    best_record = max(
        history["records"],
        key=lambda row: (row["safe_completion"], row["control_success"], row["mean_return"]),
    )
    assert history["best_step"] == best_record["step"]
    for name in (
        "mean_return", "median_return", "p10_return", "safe_completion", "control_success",
    ):
        assert history[f"best_{name}"] == best_record[name]
        assert result["evaluation"][f"best_{name}"] == best_record[name]
        assert metadata["evaluation"][f"best_{name}"] == best_record[name]
    assert (output / "best" / "model.zip").is_file()
    tracking_figure = output / "best" / "tracking.svg"
    ET.parse(tracking_figure)
    tracking_svg = tracking_figure.read_text(encoding="utf-8")
    assert "QUADRUPLE SAC BEST POLICY - VALIDATION CASES" in tracking_svg
    assert "Output: lower_tank_1_level [cm]" in tracking_svg
    assert "Applied action: pump_1_voltage [normalized_voltage]" in tracking_svg
    tracking_root = ET.fromstring(tracking_svg)
    assert [node.text for node in tracking_root.iter() if node.get("class") == "section"] == [
        "Case tracking", "Absolute performance",
    ]
    assert {node.get("data-section") for node in tracking_root.iter() if node.get("data-section")} == {
        "tracking", "summary",
    }
    assert result["best_checkpoint"] == str(
        (output / "best" / "model.zip").resolve()
    )
    assert result["best_tracking_figure"] == str(tracking_figure.resolve())
    assert result["evaluation"]["seeds"] == list(range(1_000, 1_020))
    assert result["evaluation"]["best_safe_completion"] >= 0.0
    assert result["evaluation"]["tracking_figure"] == "best/tracking.svg"
    curve_svg = (output / "training_curve.svg").read_text()
    assert "Fixed-validation episode return" in curve_svg
    assert "Fixed-validation safety and control success" in curve_svg
    assert "20 fixed validation cases" in curve_svg
    assert 'data-series="mean_return"' in curve_svg
    assert 'data-series="control_success"' in curve_svg
    curve_root = ET.fromstring(curve_svg)
    for name in ("mean_return", "p10_return", "safe_completion", "control_success"):
        plotted = [
            float(node.attrib["data-value"])
            for node in curve_root.iter()
            if node.tag.endswith("circle") and node.get("data-series") == name
        ]
        scale = 100 if name in ("safe_completion", "control_success") else 1
        assert plotted == pytest.approx([row[name] * scale for row in history["records"]])

    continuation_env = make_env("quadruple")
    continuation_evaluation_env = make_env("quadruple", randomize=True)
    try:
        continued = train(
            env=continuation_env,
            algorithm="sac",
            steps=1,
            resume_from=result["best_checkpoint"],
            evaluation_env=continuation_evaluation_env,
            evaluate_every=1,
            output=tmp_path / "continued-best",
        )
    finally:
        continuation_env.close()
        continuation_evaluation_env.close()
    assert continued["initial_steps"] == result["evaluation"]["best_step"]
    assert continued["actual_steps"] == continued["initial_steps"] + 1
    assert continued["evaluation"]["selection_order"] == history["selection_order"]
    assert "last 10%" in continued["evaluation"]["success_criterion"]


def test_training_evaluation_requires_randomized_environment(tmp_path):
    env = make_env("quadruple")
    evaluation_env = make_env("quadruple")
    try:
        with pytest.raises(ValueError, match="must use randomize=True"):
            train(
                env=env,
                algorithm="sac",
                steps=2,
                evaluation_env=evaluation_env,
                evaluate_every=1,
                output=tmp_path / "fixed-evaluation",
            )
    finally:
        evaluation_env.close()
        env.close()


def test_training_validation_variation_is_reproducible_and_preserved_on_resume(tmp_path, monkeypatch):
    module = importlib.import_module("aiogym.workflows.train")
    monkeypatch.setattr(module, "_evaluate", partial(module._evaluate, max_steps=2))
    variation = {"disturbance": True, "noise": True, "delay": True, "fault": True}
    with (
        make_env("quadruple", randomize=True, **variation) as env,
        make_env("quadruple", randomize=True, **variation) as evaluation_env,
        make_env("quadruple", randomize=True, **{**variation, "noise": False}) as changed_env,
    ):
        result = train(
            env=env, algorithm="sac", steps=1, seed=0,
            algorithm_kwargs=SMALL_POLICY, record_every=1,
            evaluation_env=evaluation_env, evaluate_every=1,
            output=tmp_path / "varied-validation",
        )
        history = json.loads((Path(result["path"]) / "evaluation_history.json").read_text())
        assert history["environment"] == result["evaluation"]["environment"]
        assert all(history["environment"][name] is not None for name in variation)
        assert history["seeds"] == list(range(1000, 1020))
        # One warm-up step leaves the policy unchanged. Repeated validation must
        # replay the same disturbances and channel samples for all 20 seeds.
        assert history["records"][0]["episodes"] == history["records"][1]["episodes"]
        for checkpoint in (result["checkpoint"], result["best_checkpoint"]):
            with zipfile.ZipFile(checkpoint) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            saved = manifest["policy"]["training_history"]["evaluation"]
            assert saved["environment"] == history["environment"]

        rejected_output = tmp_path / "changed-validation"
        with pytest.raises(ValueError, match="validation environment must match"):
            train(
                env=env, algorithm="sac", steps=1,
                evaluation_env=changed_env, evaluate_every=1,
                resume_from=result["checkpoint"], output=rejected_output,
            )
        assert not rejected_output.exists()
        continued = train(
            env=env, algorithm="sac", steps=1, record_every=1,
            evaluation_env=evaluation_env, evaluate_every=1,
            resume_from=result["checkpoint"], output=tmp_path / "continued-validation",
        )
    continued_history = json.loads(
        (Path(continued["path"]) / "evaluation_history.json").read_text()
    )
    assert continued_history["environment"] == history["environment"]
    assert continued_history["records"][:2] == history["records"]
    assert continued_history["records"][-1]["step"] == 2
    assert continued_history["records"][-1]["episodes"] == history["records"][0]["episodes"]


def test_ddpg_training_materializes_json_action_noise(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "ddpg-noise"
    kwargs = {
        "action_noise": {"std": 0.1, "type": "normal"},
        "learning_starts": 2,
        **SMALL_POLICY,
    }
    try:
        result = train(
            env=env,
            evaluate_every=None,
            algorithm="ddpg",
            steps=2,
            seed=0,
            algorithm_kwargs=kwargs,
            output=output,
        )
    finally:
        env.close()
    assert result["algorithm_kwargs"]["action_noise"] == {
        "std": 0.1,
        "type": "normal",
    }


def test_ddpg_training_accepts_ornstein_uhlenbeck_noise(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "ddpg-ou-noise"
    try:
        result = train(
            env=env,
            evaluate_every=None,
            algorithm="ddpg",
            steps=2,
            seed=0,
            algorithm_kwargs={
                "action_noise": {
                    "std": 0.1,
                    "type": "ornstein-uhlenbeck",
                },
                "learning_starts": 2,
                **SMALL_POLICY,
            },
            output=output,
        )
    finally:
        env.close()
    assert result["algorithm_kwargs"]["action_noise"]["type"] == (
        "ornstein-uhlenbeck"
    )


def test_three_tank_sac_config_matches_experimental_training_configuration():
    assert THREE_TANK_SAC_CONFIG == {
        "batch_size": 256,
        "buffer_size": 100000,
        "gamma": 0.9995,
        "learning_rate": 0.0003,
        "learning_starts": 5000,
        "n_steps": 10,
        "policy_kwargs": {"net_arch": [256, 256]},
    }


def test_training_curve_records_fixed_windows_and_can_be_replotted(tmp_path):
    env = make_env("quadruple")
    output = tmp_path / "sac"
    try:
        train(
            env=env,
            evaluate_every=None,
            algorithm="sac",
            steps=5,
            seed=0,
            algorithm_kwargs=SMALL_POLICY,
            record_every=2,
            output=output,
        )
    finally:
        env.close()

    curve_path = output / "training_curve.json"
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    assert [row["start_step"] for row in curve["records"]] == [0, 2, 4]
    assert [row["end_step"] for row in curve["records"]] == [2, 4, 5]
    assert [row["transition_count"] for row in curve["records"]] == [2, 2, 1]
    assert all(np.isfinite(row["mean_reward"]) for row in curve["records"])

    replotted = plot_training_curve(
        curve_path,
        output=tmp_path / "replotted.svg",
    )
    root = ET.parse(replotted).getroot()
    text = " ".join(element.text or "" for element in root.iter())
    assert "environment steps" in text
    assert "0" in text


def test_on_policy_training_completes(tmp_path):
    env = make_env("quadruple")
    try:
        result = train(
            env=env,
            evaluate_every=None,
            algorithm="ppo",
            steps=2,
            seed=5,
            algorithm_kwargs={
                "n_steps": 2, "batch_size": 2, "n_epochs": 1, **SMALL_POLICY,
            },
            output=tmp_path / "ppo",
        )
    finally:
        env.close()
    assert result["algorithm"] == "ppo"
    assert result["actual_steps"] == 2


@pytest.mark.parametrize("algorithm", (object, get_algorithm("sac")))
def test_algorithm_resolution_requires_builtin_name(algorithm):
    with pytest.raises(TypeError, match="built-in algorithm name"):
        get_algorithm(algorithm)


def test_checkpoint_interface_is_required_but_legacy_loading_warns(tmp_path):
    from aiogym.workflows._checkpoint import save_checkpoint

    backend = get_algorithm("sac")
    kwargs = backend.effective_kwargs(steps=2, values=SMALL_POLICY)
    env = aiogym.make_env("quadruple")
    path = tmp_path / "model.zip"
    try:
        model = backend.create(env=env, seed=8, algorithm_kwargs=kwargs)
        save_checkpoint(
            backend, model, path, env=env,
            training={"completed_steps": 2, "seed": 8, "algorithm_kwargs": kwargs, "dataset": None},
        )
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            payload = archive.read("payload.zip")
        del manifest["environment"]["policy_interface"]
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("payload.zip", payload)
        with pytest.raises(ValueError, match="missing compatibility fields.*policy_interface"):
            load_policy(path, env=env)

        manifest["schema_version"] = "aiogym.checkpoint.v2"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("payload.zip", payload)
        with pytest.warns(UserWarning, match="only legacy compatibility checks"):
            policy = load_policy(path, env=env)
        assert policy.model.seed == 8
        with pytest.warns(UserWarning, match="Channel semantics.*unverified"):
            _, _, loaded, state = load_training_checkpoint(path, env=env, algorithm="sac")
        assert loaded.seed == state["seed"] == 8
        assert state["completed_steps"] == 2
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ("ddpg", "ppo", "sac", "td3"))
def test_supported_algorithm_constructors(algorithm):
    env = make_env("quadruple")
    try:
        backend = get_algorithm(algorithm)
        kwargs = backend.effective_kwargs(steps=2, values=SMALL_POLICY)
        model = backend.create(env=env, seed=0, algorithm_kwargs=kwargs)
        assert model.action_space.shape == env.action_space.shape
    finally:
        env.close()


def test_training_rejects_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    env = make_env("quadruple")
    try:
        with pytest.raises(FileExistsError, match="non-empty"):
            train(env=env, evaluate_every=None, algorithm="sac", steps=1, output=output)
    finally:
        env.close()
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_training_rejects_benchmark_environment(tmp_path):
    env = make_env("quadruple", benchmark="tracking")
    try:
        with pytest.raises(ValueError, match="benchmark environment"):
            train(
                env=env,
                evaluate_every=None,
                algorithm="sac",
                steps=1,
                output=tmp_path / "benchmark",
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"algorithm": "unknown", "steps": 1, "seed": 0}, ValueError),
        ({"algorithm": "sac", "steps": 0, "seed": 0}, ValueError),
        ({"algorithm": "sac", "steps": True, "seed": 0}, TypeError),
        ({"algorithm": "sac", "steps": 1, "seed": -1}, ValueError),
        (
            {"algorithm": "sac", "steps": 1, "seed": 0, "record_every": 0},
            ValueError,
        ),
        (
            {"algorithm": "sac", "steps": 1, "seed": 0, "record_every": True},
            TypeError,
        ),
    ],
)
def test_training_validates_inputs(tmp_path, kwargs, error):
    env = make_env("quadruple")
    try:
        with pytest.raises(error):
            train(env=env, evaluate_every=None, output=tmp_path / "invalid", **kwargs)
    finally:
        env.close()


def test_resume_old_rule_only_selects_retained_weights(tmp_path):
    from aiogym.workflows.train import _resume_history

    best_path = tmp_path / "best" / "model.zip"
    final_path = tmp_path / "model.zip"
    best_path.parent.mkdir()
    for path, step in ((best_path, 0), (final_path, 20)):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps({
                "policy": {"training": {"completed_steps": step}},
            }))
    records = [{"step": step, "episodes": [{
        "seed": 1000, "return": reward, "length": 100,
        "terminated": False, "truncated": True,
        "episode_spec": {"horizon": 100}, "runtime_variation": {},
        "metrics": {"settling_time": settling, "constraint_violations": 0},
    }]} for step, reward, settling in ((0, -1, 100), (10, 0, 0), (20, -3, 0))]
    curve = {
        "schema_version": "aiogym.training_curve.v2", "record_every": 20,
        "initial_steps": 0, "actual_steps": 20, "episodes": [],
        "records": [{"start_step": 0, "end_step": 20, "transition_count": 20,
                     "mean_reward": 0, "reward_std": 0, "minimum_reward": 0,
                     "maximum_reward": 0, "completed_episodes": 0,
                     "terminated_episodes": 0, "truncated_episodes": 0}],
    }
    training = {
        "completed_steps": 20,
        "history": {"curve": curve, "best_checkpoint": str(best_path),
                    "evaluation": {"schema_version": "aiogym.training_evaluation.v3",
                                   "seeds": [1000], "best_step": 0,
                                   "settling_window": 30, "records": records}},
    }
    with make_env("heater", randomize=True) as evaluation_env:
        restored = _resume_history(final_path, training, evaluation_env=evaluation_env)
    with make_env("heater", randomize=True, noise=True) as evaluation_env:
        with pytest.raises(ValueError, match="history has no environment metadata"):
            _resume_history(final_path, training, evaluation_env=evaluation_env)
    assert restored["evaluation"]["best_step"] == 20
    assert restored["best_checkpoint"] == final_path
    assert restored["evaluation"]["records"][1]["selection_eligible"] is False
    assert "last 10%" in restored["evaluation"]["success_criterion"]

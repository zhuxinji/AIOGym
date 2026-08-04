from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from aiogym._internal.serialization import file_sha256
from aiogym.evaluation.artifact.plotting import _resolved_rollouts
from aiogym.evaluation.artifact.svg import plot_learning_curve
from aiogym.evaluation.artifact.tables import _tracking_rollout_groups
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.plan import resolve_training_plan
from scripts.experiments.capture_runtime_environment import (
    capture_runtime_environment,
)
from scripts.experiments.make_review_bundle import make_review_bundle
from scripts.experiments.plot_learning_curves import plot_learning_curves
from scripts.experiments.preflight_protocol import preflight_protocol
from scripts.experiments.review_training_run import review_training_run


ROOT = Path(__file__).resolve().parents[2]


def test_quadruple_pilot_preflight_has_no_training_or_test_rollout():
    report = preflight_protocol(
        ROOT / "configs/experiments/quadruple/sac-pilot-v2.json"
    )

    assert report["passed"] is True
    assert report["track_id"] == "quadruple-regulation-generalist-v2"
    assert report["unique_training_episode_specs"] >= 2
    assert report["test_rollouts"] == 0
    assert report["optimizer_updates"] == 0
    assert all(
        value not in {float("inf"), float("-inf")}
        for value in report["observation_space"]["low"]
    )


def test_cascade_pilot_preflight_freezes_hidden_disturbance_contract():
    report = preflight_protocol(
        ROOT / "configs/experiments/cascade/sac-pilot-v2.json"
    )

    assert report["passed"] is True
    assert report["track_id"] == "cascade-regulation-generalist-v2"
    assert report["unique_training_episode_specs"] >= 2
    assert report["policy_contract"]["normalize_observations"] is True
    assert report["policy_contract"]["disturbance_obs"] is False
    assert report["test_rollouts"] == 0
    assert report["optimizer_updates"] == 0
    assert report["training_distribution_id"] == (
        "cascade-regulation-training-l2-v2"
    )
    assert len(report["training_distribution_hash"]) == 64
    modes = {
        row["difficulty_tags"][-1]
        for row in report["training_episode_samples"]
    }
    assert modes == {"commissioning", "temperature-step"}
    assert {
        row["reference_event_count"]
        for row in report["training_episode_samples"]
    } == {3, 5}
    assert {
        row["episode_steps"]
        for row in report["training_episode_samples"]
    } == {2640}
    assert all(
        value not in {float("inf"), float("-inf")}
        for value in report["observation_space"]["low"]
    )


def test_cascade_retry_changes_only_learning_rate_and_output_identity():
    baseline = json.loads(
        (ROOT / "configs/experiments/cascade/sac-pilot-v2.json").read_text(
            encoding="utf-8"
        )
    )
    retry = json.loads(
        (
            ROOT
            / "configs/experiments/cascade/sac-pilot-lr1e4-v2.json"
        ).read_text(encoding="utf-8")
    )

    baseline_learning_rate = baseline["algorithm"].pop("learning_rate")
    retry_learning_rate = retry["algorithm"].pop("learning_rate")
    baseline.pop("output")
    retry.pop("output")

    assert baseline_learning_rate == 3e-4
    assert retry_learning_rate == 1e-4
    assert retry == baseline


def test_cascade_learning_rate_midpoint_changes_only_controlled_factor():
    lower = json.loads(
        (
            ROOT
            / "configs/experiments/cascade/sac-pilot-lr1e4-v2.json"
        ).read_text(encoding="utf-8")
    )
    midpoint = json.loads(
        (
            ROOT
            / "configs/experiments/cascade/sac-pilot-lr2e4-v2.json"
        ).read_text(encoding="utf-8")
    )

    lower_learning_rate = lower["algorithm"].pop("learning_rate")
    midpoint_learning_rate = midpoint["algorithm"].pop("learning_rate")
    lower.pop("output")
    midpoint.pop("output")

    assert lower_learning_rate == 1e-4
    assert midpoint_learning_rate == 2e-4
    assert midpoint == lower


def test_case_conditioned_retry_keeps_safe_learning_rate_config_frozen():
    lower = json.loads(
        (
            ROOT
            / "configs/experiments/cascade/sac-pilot-lr1e4-v2.json"
        ).read_text(encoding="utf-8")
    )
    repaired = json.loads(
        (
            ROOT
            / "configs/experiments/cascade/sac-pilot-case-conditioned-v2.json"
        ).read_text(encoding="utf-8")
    )

    lower.pop("output")
    repaired.pop("output")
    assert repaired == lower


def test_early_diagnostic_changes_only_observation_window_and_output():
    pilot = json.loads(
        (
            ROOT
            / "configs/experiments/cascade/sac-pilot-case-conditioned-v2.json"
        ).read_text(encoding="utf-8")
    )
    diagnostic = json.loads(
        (
            ROOT
            / "configs/experiments/cascade/sac-early-diagnostics-v2.json"
        ).read_text(encoding="utf-8")
    )

    assert diagnostic["budget"]["value"] == 60000
    assert diagnostic["evaluation"]["every_transitions"] == 2000
    assert diagnostic["checkpointing"]["every_transitions"] == 10000
    pilot.pop("budget")
    diagnostic.pop("budget")
    pilot.pop("evaluation")
    diagnostic.pop("evaluation")
    pilot.pop("checkpointing")
    diagnostic.pop("checkpointing")
    pilot.pop("output")
    diagnostic.pop("output")
    assert diagnostic == pilot


def test_runtime_environment_capture_is_read_only_and_complete():
    report = capture_runtime_environment()

    assert report["python"]["version"]
    assert report["platform"]["system"]
    assert "numpy" in report["dependency_versions"]
    assert "num_threads" in report["torch_runtime"]
    assert "commit" in report["git"]
    assert "dirty" in report["git"]


def test_review_plot_and_bundle_use_text_artifacts_only(tmp_path):
    summary = _synthetic_review_tree(tmp_path)

    review = review_training_run(summary)
    assert review["passed"] is True
    assert review["runs"][0]["learning_curve"][
        "final_checkpoint_participated"
    ] is True
    review_json = tmp_path / "review.json"
    review_json.write_text(json.dumps(review), encoding="utf-8")
    review_md = tmp_path / "review.md"
    review_md.write_text("# Review\n", encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(capture_runtime_environment()),
        encoding="utf-8",
    )
    log = tmp_path / "pilot.log"
    log.write_text("line\n" * 400, encoding="utf-8")

    plot = plot_learning_curves(summary, tmp_path / "curves.svg")
    assert plot["series"] == 1
    assert plot["points"] == 2
    assert plot["panels"] == [
        "aggregate regulation_cost_rate",
        "minimum-phase:validation-offset-v2",
        "nonminimum-phase:validation-offset-v2",
    ]
    plot_svg = (tmp_path / "curves.svg").read_text(encoding="utf-8")
    assert "Validation learning curves by Case" in plot_svg
    assert "markers are validation checkpoints" in plot_svg
    bundle_path = tmp_path / "review-bundle.zip"
    bundle = make_review_bundle(
        summary,
        bundle_path,
        runtime_report=runtime,
        review_json=review_json,
        review_markdown=review_md,
        log=log,
    )
    replacement = make_review_bundle(
        summary,
        bundle_path,
        runtime_report=runtime,
        review_json=review_json,
        review_markdown=review_md,
        log=log,
        overwrite=True,
    )
    assert bundle_path.is_file()
    assert bundle["included"]
    assert replacement["included"]
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        assert "MANIFEST.json" in names
        assert "logs/training.tail.log" in names
        assert not any(name.endswith((".zip", ".pt", ".pkl")) for name in names)


def test_review_refuses_test_split_artifact(tmp_path):
    source = tmp_path / "test-result.json"
    source.write_text(
        json.dumps({"split": "test", "runs": [{}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refuses test-split"):
        review_training_run(source)


def test_review_allows_static_test_protocol_metadata(tmp_path):
    summary = _synthetic_review_tree(tmp_path)
    run = json.loads(summary.read_text(encoding="utf-8"))["runs"][0]
    benchmark_path = Path(run["artifact_dir"]) / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["evaluation"] = {
        "resolved_cases": {
            "test": [
                {
                    "split": "test",
                    "case_id": "static-protocol-only",
                    "resolved_case_hash": "9" * 64,
                    "profile": {},
                }
            ]
        }
    }
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

    assert review_training_run(summary)["passed"] is True


def test_tracking_rollouts_keep_case_id_when_legacy_case_is_missing():
    payload = {
        "scenario": "quadruple",
        "goal": "regulation",
        "rollouts": [
            {"case_id": "minimum-phase:validation-offset-v2"},
            {"case_id": "nonminimum-phase:validation-offset-v2"},
        ],
    }

    rollouts = _resolved_rollouts(payload)
    groups = _tracking_rollout_groups(rollouts)

    assert set(groups) == {
        ("quadruple", "minimum-phase:validation-offset-v2"),
        ("quadruple", "nonminimum-phase:validation-offset-v2"),
    }


def test_learning_curve_plots_cost_by_case_without_numeric_metadata(tmp_path):
    output = tmp_path / "learning.svg"
    plot_learning_curve(
        [
            {
                "step": 0,
                "metric": "regulation_cost_rate",
                "metric_value": 1.0,
                "case_values": [0.5, 1.5],
                "case_count": 2,
                "official_score": 0.0,
                "ranking_epsilon": 1e-12,
            },
            {
                "step": 100,
                "metric": "regulation_cost_rate",
                "metric_value": 0.01,
                "case_values": [0.005, 0.015],
                "case_count": 2,
                "official_score": 1e-12,
                "ranking_epsilon": 1e-12,
            },
        ],
        str(output),
        "quadruple",
        case_ids=["minimum-phase", "nonminimum-phase"],
    )

    svg = output.read_text(encoding="utf-8")
    assert "aggregate regulation cost rate" in svg
    assert "minimum-phase" in svg
    assert "nonminimum-phase" in svg
    assert "log scale" in svg
    assert "<circle" in svg
    assert "case count" not in svg
    assert "official score" not in svg
    assert "ranking epsilon" not in svg


def _synthetic_review_tree(tmp_path: Path) -> Path:
    raw = {
        "schema_version": "aiogym.rl_training_config.v3",
        "track_id": "quadruple-regulation-generalist-v2",
        "algorithm_id": "sac",
        "training_seed": 0,
        "budget": {"unit": "environment_transitions", "value": 100},
        "n_envs": 1,
        "algorithm": {"vector_backend": "dummy"},
        "output": {
            "directory": str(tmp_path),
            "name": "pilot-seed0",
        },
        "validation_seeds": [5000],
    }
    plan = resolve_training_plan(RLTrainingConfig.from_mapping(raw))
    plan.artifact_dir.mkdir(parents=True)
    plan.resolved_config_path.write_text(
        json.dumps(plan.config.as_dict()),
        encoding="utf-8",
    )
    plan.policy_path.write_bytes(b"policy")
    benchmark = {
        "training": {
            "training_config_hash": plan.config.config_hash,
            "track_hash": plan.track.track_hash,
            "device": "cpu",
            "runtime": {"steps_per_second": 10.0},
            "training_episode_specs": [
                {"episode_spec_hash": "e" * 64},
                {"episode_spec_hash": "f" * 64},
            ],
            "unique_training_episode_specs": 2,
        },
        "learning_curve": [
            {
                "timesteps": 50,
                "metric": "regulation_cost_rate",
                "metric_value": 2.0,
                "case_values": [1.0, 3.0],
            },
            {
                "timesteps": 100,
                "metric": "regulation_cost_rate",
                "metric_value": 1.0,
                "case_values": [0.5, 1.5],
            },
        ],
        "track_evaluation": {
            "results": [
                {"case_id": "minimum-phase:validation-offset-v2"},
                {"case_id": "nonminimum-phase:validation-offset-v2"},
            ]
        },
    }
    (plan.artifact_dir / "benchmark.json").write_text(
        json.dumps(benchmark),
        encoding="utf-8",
    )
    (plan.artifact_dir / "report.md").write_text(
        "# Benchmark\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": "aiogym.run_result.v3",
        "config_hash": plan.config.config_hash,
        "track_id": plan.track.id,
        "track_hash": plan.track.track_hash,
        "algorithm_id": "sac",
        "training_seed": 0,
        "output_dir": str(plan.output_dir),
        "policy_path": str(plan.policy_path),
        "artifact_dir": str(plan.artifact_dir),
        "resolved_config_path": str(plan.resolved_config_path),
        "validation_plan_hash": plan.validation_plan.plan_hash,
        "policy_sha256": file_sha256(plan.policy_path),
        "backend": {
            "final_step": 100,
            "checkpoint_selection": "best-validation",
            "selected_checkpoint_step": 100,
            "runtime": {"steps_per_second": 10.0},
        },
        "validation": {
            "split": "validation",
            "results": [
                {
                    "case_id": "minimum-phase:validation-offset-v2",
                    "resolved_case_hash": "1" * 64,
                    "ranking_eligible": True,
                },
                {
                    "case_id": "nonminimum-phase:validation-offset-v2",
                    "resolved_case_hash": "2" * 64,
                    "ranking_eligible": True,
                },
            ],
            "aggregate": {
                "metric": "regulation_cost_rate",
                "metric_direction": "minimize",
                "metric_value": 1.5,
                "case_values": [1.0, 2.0],
                "official_score": 80.0,
                "ranking_eligible": True,
            },
        },
    }
    run_path = plan.resolved_config_path.with_name("pilot-seed0.run-result.json")
    run_path.write_text(json.dumps(run), encoding="utf-8")
    summary = tmp_path / "pilot.multi-seed.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "aiogym.multi_seed_run.v1",
                "seeds": [0],
                "runs": [run],
                "training_seed_statistics": {"training_seeds": [0]},
            }
        ),
        encoding="utf-8",
    )
    return summary

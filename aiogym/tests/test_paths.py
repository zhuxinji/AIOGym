"""Generated-output path defaults and overrides."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aiogym._internal.paths import RUNS_DIR_ENV, run_path, runs_dir
from aiogym.cli.benchmark import artifact_dir_for as cli_artifact_dir_for
from aiogym.evaluation.suite import artifact_dir_for
from aiogym.evaluation import run_benchmark
from aiogym.rl.train_rlpd import output_base_for


def test_runs_dir_defaults_to_working_directory_runs(monkeypatch):
    monkeypatch.delenv(RUNS_DIR_ENV, raising=False)

    assert str(runs_dir()) == "runs"
    assert str(run_path("rl", "sb3")) == "runs/rl/sb3"
    args = SimpleNamespace(artifact_dir=None, overwrite=True, task="minimum-phase")
    suite = {"name": "quadruple"}
    cases = [{"scenario": "quadruple", "task": "minimum-phase"}]
    assert cli_artifact_dir_for(
        args,
        suite,
        cases,
    ) == "runs/quadruple/minimum-phase"
    args.overwrite = False
    assert cli_artifact_dir_for(
        args,
        suite,
        cases,
        run_id="fixed",
    ) == "runs/quadruple/minimum-phase_fixed"
    assert artifact_dir_for("example") == "runs/example_suite"
    assert artifact_dir_for("example", overwrite=False, run_id="fixed") == (
        "runs/example_suite_fixed"
    )
    args = SimpleNamespace(out=None, scenario="cstr")
    assert output_base_for(args, "fixed") == "runs/rl/rlpd/cstr_fixed"


def test_runs_dir_environment_override_applies_only_to_defaults(monkeypatch, tmp_path):
    custom_root = tmp_path / "generated"
    monkeypatch.setenv(RUNS_DIR_ENV, str(custom_root))

    assert runs_dir() == custom_root
    assert run_path("model_metadata") == custom_root / "model_metadata"
    args = SimpleNamespace(artifact_dir=None, overwrite=True, task="default")
    assert cli_artifact_dir_for(
        args,
        {"name": "heater"},
        [{"scenario": "heater", "task": "default"}],
    ) == str(custom_root / "heater" / "default")
    assert artifact_dir_for("example") == str(
        custom_root / "example_suite"
    )
    assert artifact_dir_for(
        "example", overwrite=False, run_id="fixed"
    ) == str(
        custom_root / "example_suite_fixed"
    )
    args = SimpleNamespace(out=None, scenario="heater")
    assert output_base_for(args, "fixed") == str(
        custom_root / "rl" / "rlpd" / "heater_fixed"
    )

    assert artifact_dir_for("example", artifact_dir="explicit/suite") == (
        "explicit/suite"
    )
    assert artifact_dir_for(
        "example",
        artifact_dir="explicit/suite",
        overwrite=False,
        run_id="fixed",
    ) == "explicit/suite_fixed"
    args.out = "explicit/rlpd"
    assert output_base_for(args, "ignored") == "explicit/rlpd"


def test_empty_runs_dir_environment_uses_default(monkeypatch):
    monkeypatch.setenv(RUNS_DIR_ENV, "")

    assert str(runs_dir()) == "runs"


def test_public_benchmark_uses_configured_default_root(monkeypatch, tmp_path):
    custom_root = tmp_path / "benchmark-output"
    monkeypatch.setenv(RUNS_DIR_ENV, str(custom_root))
    expected = custom_root / "cstr" / "default"
    stale_rollout = expected / "rollouts" / "rollouts.json"
    stale_rollout.parent.mkdir(parents=True)
    stale_rollout.write_text("stale")
    user_file = expected / "review-notes.md"
    user_file.write_text("keep")

    payload = run_benchmark({
        "scenario": "cstr",
        "objective": "tracking",
        "controllers": ["pid"],
        "seeds": [3],
        "episode_steps": 1,
        "include_episodes": False,
    })

    assert payload["run_dir"] == str(expected)
    assert (expected / "benchmark.json").is_file()
    assert not stale_rollout.exists()
    assert user_file.read_text() == "keep"


def test_public_benchmark_explicit_output_dir_overwrites_managed_artifacts(
    tmp_path,
):
    explicit = tmp_path / "explicit"
    stale_rollout = explicit / "rollouts" / "rollouts.json"
    stale_rollout.parent.mkdir(parents=True)
    stale_rollout.write_text("caller-owned")
    user_file = explicit / "review-notes.md"
    user_file.write_text("keep")

    run_benchmark({
        "scenario": "cstr",
        "objective": "tracking",
        "controllers": ["pid"],
        "seeds": [3],
        "episode_steps": 1,
        "include_episodes": False,
        "output_dir": str(explicit),
    })

    assert not stale_rollout.exists()
    assert user_file.read_text() == "keep"


def test_public_benchmark_no_overwrite_uses_timestamped_sibling(tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    marker = explicit / "keep.txt"
    marker.write_text("original")

    payload = run_benchmark({
        "scenario": "cstr",
        "objective": "tracking",
        "controllers": ["pid"],
        "seeds": [3],
        "episode_steps": 1,
        "include_episodes": False,
        "output_dir": str(explicit),
        "overwrite": False,
    })

    actual = Path(payload["run_dir"])
    assert actual.parent == explicit.parent
    assert actual.name.startswith("explicit_")
    assert actual != explicit
    assert (actual / "benchmark.json").is_file()
    assert marker.read_text() == "original"

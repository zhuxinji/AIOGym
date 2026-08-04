"""Parity tests for raw checkpoints and self-describing run inputs."""
from __future__ import annotations

import hashlib
import json

import pytest

from aiogym.benchmarks.tracks.registry import load_track
from aiogym.cli import benchmark as benchmark_cli
from aiogym.cli import evaluate as evaluate_cli
from aiogym.rl.runner import RUN_RESULT_SCHEMA_VERSION


TRACK_ID = "quadruple-regulation-generalist-v1"


def _run_fixture(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    digest = hashlib.sha256(b"policy").hexdigest()
    track = load_track(TRACK_ID)
    manifest = tmp_path / "training.run-result.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": RUN_RESULT_SCHEMA_VERSION,
                "track_id": track.id,
                "track_hash": track.track_hash,
                "algorithm_id": "bc",
                "policy_path": str(checkpoint),
                "policy_sha256": digest,
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest, checkpoint, digest, track


def _evaluation_result(track, *, benchmark=False):
    aggregate = {
        "official_score": 1.0,
        "metric": "regulation_cost_rate",
        "metric_value": 1.0,
        "ranking_eligible": True,
    }
    if not benchmark:
        aggregate = {"official_score": 1.0}
    return {
        "track_id": track.id,
        "track_hash": track.track_hash,
        "split": "validation",
        "seed_namespace": "validation",
        "base_seeds": [5000],
        "case_count": 1,
        "aggregate": aggregate,
    }


def test_evaluate_run_and_raw_checkpoint_resolve_equivalently(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest, checkpoint, digest, track = _run_fixture(tmp_path)
    loaded = []
    evaluated = []
    monkeypatch.setattr(
        evaluate_cli,
        "load_policy_checkpoint",
        lambda spec, **kwargs: loaded.append(spec) or object(),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "evaluate_policy_on_track",
        lambda controller, selected, **kwargs: evaluated.append(
            (selected.id, kwargs)
        )
        or _evaluation_result(track),
    )

    assert evaluate_cli.main([str(manifest)]) == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert evaluate_cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--track",
            "quadruple",
            "--algorithm",
            "bc",
            "--sha256",
            digest,
        ]
    ) == 0
    raw_payload = json.loads(capsys.readouterr().out)

    assert loaded[0] == loaded[1]
    assert evaluated[0] == evaluated[1]
    assert run_payload["aggregate"] == raw_payload["aggregate"]
    assert run_payload["run_reference"] == str(manifest)
    assert raw_payload["run_reference"] is None


def test_evaluate_run_rejects_raw_checkpoint_options(tmp_path):
    manifest, _, _, _ = _run_fixture(tmp_path)

    with pytest.raises(SystemExit):
        evaluate_cli.main([str(manifest), "--track", "quadruple"])


def test_benchmark_run_and_raw_checkpoint_resolve_equivalently(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest, checkpoint, digest, track = _run_fixture(tmp_path)
    loaded = []
    monkeypatch.setattr(
        benchmark_cli,
        "load_policy_checkpoint",
        lambda spec, **kwargs: loaded.append(spec) or object(),
    )
    monkeypatch.setattr(
        benchmark_cli,
        "make_controller",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark_cli,
        "evaluate_policy_on_track",
        lambda *args, **kwargs: _evaluation_result(track, benchmark=True),
    )
    run_output = tmp_path / "run.json"
    raw_output = tmp_path / "raw.json"

    assert benchmark_cli.main(
        [
            "--run",
            str(manifest),
            "--controllers",
            "pid",
            "--output",
            str(run_output),
        ]
    ) == 0
    capsys.readouterr()
    assert benchmark_cli.main(
        [
            "quadruple",
            "--controllers",
            "pid",
            "--checkpoint",
            str(checkpoint),
            "--algorithm",
            "bc",
            "--sha256",
            digest,
            "--name",
            "bc",
            "--output",
            str(raw_output),
        ]
    ) == 0
    capsys.readouterr()

    assert loaded[0] == loaded[1]
    assert json.loads(run_output.read_text()) == json.loads(
        raw_output.read_text()
    )


def test_benchmark_run_rejects_mismatch_and_raw_checkpoint_options(tmp_path):
    manifest, _, _, _ = _run_fixture(tmp_path)

    with pytest.raises(SystemExit):
        benchmark_cli.main(
            ["cascade", "--run", str(manifest), "--controllers", "pid"]
        )
    with pytest.raises(SystemExit):
        benchmark_cli.main(
            [
                "--run",
                str(manifest),
                "--controllers",
                "pid",
                "--algorithm",
                "bc",
            ]
        )

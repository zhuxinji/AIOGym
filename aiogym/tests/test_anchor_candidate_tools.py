"""Static tests for v2 anchor candidate generation and review tooling."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import calibrate_official_anchors as calibration_script
from scripts.experiments.review_anchor_candidates import (
    review_anchor_candidate,
    review_anchor_candidates,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v2_anchor_suite_is_selectable_without_touching_builtins(
    tmp_path,
    monkeypatch,
):
    calls = []

    def calibrate(track, **kwargs):
        calls.append((track.id, kwargs))
        return {
            "track_id": track.id,
            "id": kwargs["anchor_id"],
        }

    monkeypatch.setattr(
        calibration_script,
        "calibrate_anchor_manifest",
        calibrate,
    )
    builtin_dir = ROOT / "aiogym/benchmarks/anchors/builtin"
    builtin_v3_before = {
        path.name: path.read_bytes()
        for path in builtin_dir.glob("*-anchors-v3.json")
    }
    outputs = calibration_script.generate_official_anchor_candidates(
        tmp_path,
        suite="v2-regulation",
    )
    assert [path.name for path in outputs] == [
        "quadruple-regulation-anchors-v3.json",
        "cascade-regulation-anchors-v3.json",
        "cascade-recirculating-regulation-anchors-v3.json",
    ]
    assert [row[0] for row in calls] == [
        "quadruple-regulation-generalist-v2",
        "cascade-regulation-generalist-v2",
        "cascade-recirculating-regulation-generalist-v2",
    ]
    assert all(path.parent == tmp_path for path in outputs)
    assert {
        path.name: path.read_bytes()
        for path in builtin_dir.glob("*-anchors-v3.json")
    } == builtin_v3_before


def test_anchor_calibration_refuses_existing_candidate_before_work(
    tmp_path,
    monkeypatch,
):
    existing = tmp_path / "quadruple-regulation-anchors-v3.json"
    existing.write_text("{}\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(
        calibration_script,
        "calibrate_anchor_manifest",
        lambda *args, **kwargs: called.append(True),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        calibration_script.generate_official_anchor_candidates(
            tmp_path,
            suite="v2-regulation",
        )
    assert called == []


def test_anchor_calibration_rejects_unknown_suite(tmp_path):
    with pytest.raises(ValueError, match="unknown anchor calibration suite"):
        calibration_script.generate_official_anchor_candidates(
            tmp_path,
            suite="unknown",
        )


def test_cascade_v2_anchor_can_be_retried_without_quadruple(
    tmp_path,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        calibration_script,
        "calibrate_anchor_manifest",
        lambda track, **kwargs: calls.append(track.id)
        or {"track_id": track.id, "id": kwargs["anchor_id"]},
    )
    outputs = calibration_script.generate_official_anchor_candidates(
        tmp_path,
        suite="v2-cascade",
    )
    assert [path.name for path in outputs] == [
        "cascade-regulation-anchors-v3.json"
    ]
    assert calls == ["cascade-regulation-generalist-v2"]


def test_review_accepts_complete_hash_protected_existing_manifest(tmp_path):
    source = (
        ROOT
        / "aiogym/benchmarks/anchors/builtin"
        / "quadruple-regulation-anchors-v2.json"
    )
    candidate = tmp_path / source.name
    candidate.write_bytes(source.read_bytes())
    row = review_anchor_candidate(candidate)
    report = review_anchor_candidates(tmp_path)
    assert row["passed"] is True
    assert row["artifact_hash_valid"] is True
    assert row["schema_valid"] is True
    assert row["missing_case_hashes"] == []
    assert row["extra_case_hashes"] == []
    assert row["dependency_versions"]
    assert all(case["reference_eligible"] for case in row["cases"])
    assert report["passed"] is True
    assert report["candidate_count"] == 1


def test_review_reports_tampered_artifact_hash(tmp_path):
    source = (
        ROOT
        / "aiogym/benchmarks/anchors/builtin"
        / "quadruple-regulation-anchors-v2.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["artifact_hash"] = "0" * 64
    candidate = tmp_path / source.name
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    row = review_anchor_candidate(candidate)
    assert row["passed"] is False
    assert row["artifact_hash_valid"] is False
    assert "artifact hash mismatch" in row["failure_reasons"]

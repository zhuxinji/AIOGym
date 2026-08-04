from __future__ import annotations

from copy import deepcopy

import pytest

from aiogym.benchmarks.anchors import (
    anchor_artifact_hash,
    load_anchor_set,
)
from aiogym.benchmarks.anchors.quality import (
    anchor_quality_statistics,
    validate_anchor_gap,
)
from aiogym.benchmarks.anchors.audit import audit_builtin_anchors
from aiogym.benchmarks.ranking import fixed_anchor_score
from aiogym.benchmarks.tracks.registry import list_tracks, load_track


LEGACY_DEGENERATE_ANCHORS = (
    ("cascade-economic:continuous-benchmark", -21.3865040389, -21.3865015138),
    ("cascade:disturbance-rejection:nominal", -7.5888e-17, -1.8793e-19),
    (
        "cascade-recirculating:disturbance-rejection:nominal",
        -1.5032e-16,
        -8.3063e-20,
    ),
    ("cascade-recirculating:commissioning", -4.2199e-17, -4.7321e-20),
    ("quadruple:disturbance-rejection:nominal", -2.1298e-15, -9.8283e-20),
)


@pytest.mark.parametrize(("case_id", "bad", "reference"), LEGACY_DEGENERATE_ANCHORS)
def test_observed_v1_anchor_regressions_fail_v2_policy(case_id, bad, reference):
    with pytest.raises(ValueError, match="degenerate"):
        validate_anchor_gap(bad, reference)


def test_loader_rejects_numerically_degenerate_anchor():
    payload = _anchor_payload(
        anchor_quality_statistics([0.0] * 5, [1e-12] * 5)
    )
    payload["artifact_hash"] = anchor_artifact_hash(payload)

    with pytest.raises(ValueError, match="degenerate ranking anchor"):
        load_anchor_set(payload)


def test_fixed_anchor_score_rejects_tiny_denominator():
    with pytest.raises(ValueError, match="degenerate"):
        fixed_anchor_score(
            5e-13,
            bad_utility=0.0,
            reference_utility=1e-12,
        )


def test_anchor_quality_statistics_are_hash_protected():
    track = load_track("quadruple-regulation-generalist-v1")
    anchors = load_anchor_set(track.ranking_declaration["anchor_id"], track=track)
    payload = anchors.metadata()
    case = next(iter(payload["cases"].values()))
    case["quality"]["bad_samples"][0] -= 1.0
    payload["artifact_hash"] = anchor_artifact_hash(payload)

    with pytest.raises(ValueError, match="statistics do not match"):
        load_anchor_set(payload)


@pytest.mark.parametrize(
    "track_id",
    tuple(
        track_id
        for track_id in list_tracks()
        if load_track(track_id).ranking_declaration.get("anchor_id")
    ),
)
def test_official_track_anchor_quality(track_id):
    track = load_track(track_id)
    anchors = load_anchor_set(track.ranking_declaration["anchor_id"], track=track)

    assert anchors.cases
    assert all(row.quality["passed"] for row in anchors.cases.values())
    for row in anchors.cases.values():
        midpoint = 0.5 * (row.bad_utility + row.reference_utility)
        assert fixed_anchor_score(
            midpoint,
            bad_utility=row.bad_utility,
            reference_utility=row.reference_utility,
        ) == pytest.approx(50.0)


def test_anchor_audit_covers_every_builtin_fixed_anchor_case():
    rows = audit_builtin_anchors()

    assert len(rows) == 33
    assert all(row["quality_passed"] for row in rows)
    assert all(row["absolute_gap"] >= 1e-8 for row in rows)
    assert all(row["relative_gap"] >= 1e-4 for row in rows)


def test_nominal_shifted_pairs_share_context_except_declared_disturbance():
    for track_id in (
        "cascade-regulation-generalist-v1",
        "cascade-recirculating-regulation-generalist-v1",
        "quadruple-regulation-generalist-v1",
    ):
        cases = load_track(track_id).resolved_cases("test")
        pair = [case for case in cases if case.pair_id is not None]
        assert {case.condition for case in pair} == {"nominal", "shifted"}
        nominal = next(case for case in pair if case.condition == "nominal")
        shifted = next(case for case in pair if case.condition == "shifted")
        nominal_profile = deepcopy(dict(nominal.profile))
        shifted_profile = deepcopy(dict(shifted.profile))
        assert nominal_profile.pop("disturbances") == []
        assert shifted_profile.pop("disturbances")
        assert nominal_profile == shifted_profile
        assert nominal.pair_id == shifted.pair_id
        assert nominal.base_case_id == shifted.base_case_id


def test_cascade_economic_reference_has_meaningful_effect_size():
    row = next(
        row
        for row in audit_builtin_anchors()
        if row["track_id"] == "cascade-economic-specialist-v1"
    )

    assert row["absolute_gap"] > 1.0
    assert row["relative_gap"] > 0.5


def _anchor_payload(quality):
    case_hash = "a" * 64
    payload = {
        "schema_version": "aiogym.ranking_anchors.v2",
        "id": "synthetic-anchors-v2",
        "track_id": "synthetic-track-v1",
        "track_hash": "b" * 64,
        "ranking_spec_id": "synthetic-ranking-v1",
        "goal": "regulation",
        "bad_controller": {"id": "hold"},
        "reference_controller": {"id": "pid"},
        "calibration": {
            "generator": "synthetic-test",
            "paired_seed_evaluation": True,
            "dependency_versions": {},
        },
        "evaluation_seeds": [0, 1, 2, 3, 4],
        "cases": {
            case_hash: {
                "case_id": "synthetic",
                "resolved_case_hash": case_hash,
                "bad_utility": quality["bad_mean"],
                "reference_utility": quality["reference_mean"],
                "bad_controller": {"id": "hold"},
                "reference_controller": {"id": "pid"},
                "quality": deepcopy(quality),
            }
        },
    }
    return payload

from __future__ import annotations

from copy import deepcopy

import pytest

from aiogym.benchmarks import TrackSpec, evaluate_policy_on_track, load_track
from aiogym.evaluation.safety_gate import SafetyGateSpec


def test_builtin_track_binds_declared_gate():
    track = load_track("quadruple-regulation-generalist-v1")
    visited = []

    class FakeEnv:
        control_dt = 1.0
        episode_steps = 2

        def __init__(self, case):
            self.case = case

        def close(self):
            pass

    def evaluate(_controller, env, **kwargs):
        visited.append(kwargs["safety_gate_spec"])
        return {
            "metric": "regulation_cost",
            "metric_direction": "minimize",
            "regulation_cost": 2.0,
            "ranking_eligible": True,
        }

    output = evaluate_policy_on_track(
        object(),
        track,
        base_seeds=[10],
        env_factory=FakeEnv,
        evaluate_fn=evaluate,
    )
    assert visited
    assert all(spec == track.safety_gate_spec() for spec in visited)
    assert output["aggregate"]["official_score"] >= 0.0


def test_builtin_track_rejects_gate_override():
    track = load_track("quadruple-regulation-generalist-v1")
    with pytest.raises(ValueError, match="cannot be overridden"):
        evaluate_policy_on_track(
            object(),
            track,
            safety_gate_spec=SafetyGateSpec(mode="recovery"),
        )


def test_unknown_gate_rejected_when_track_is_loaded():
    declaration = deepcopy(
        load_track(
            "quadruple-regulation-generalist-v1",
            validate_policy_contract=False,
        ).declaration
    )
    declaration["id"] = "unknown-gate-track-v1"
    declaration["ranking"]["safety_gate"] = "missing-gate-v1"
    with pytest.raises(ValueError, match="unknown benchmark safety gate"):
        TrackSpec(declaration)

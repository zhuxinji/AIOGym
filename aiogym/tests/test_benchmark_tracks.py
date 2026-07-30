from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from aiogym.benchmarks import (
    TrackSpec,
    evaluate_policy_on_track,
    load_track,
)
from aiogym.rl.train_sb3 import run_name_for
from aiogym.rl.training_artifacts import (
    RL_ARTIFACT_SCHEMA_VERSION,
    rl_payload,
)
from aiogym.rl.training_config import configure_training_track


def test_track_has_no_objective_cartesian_expansion():
    track = load_track(
        "quadruple-regulation-generalist-v1"
    )
    assert track.goal == "regulation"
    assert track.reward_spec_id == "regulation-v1"
    assert [case.case_id for case in track.resolved_cases("training")] == [
        "minimum-phase",
        "nonminimum-phase",
    ]
    assert "objective" not in track.declaration
    assert "objectives" not in track.declaration


def test_track_rejects_incompatible_control_dt():
    declaration = _track_declaration(
        scenario="cascade",
        training_cases=["commissioning", "safety-recovery"],
        validation_cases=["commissioning"],
        test_cases=["commissioning"],
        control_dt=0.5,
    )
    track = TrackSpec(declaration)
    with pytest.raises(ValueError, match="control_dt"):
        track.validate_policy_contract()


def test_track_rejects_observation_contract_mismatch():
    shifted = {
        "base_case": "minimum-phase",
        "variant_id": "observed-disturbance",
        "overrides": {
            "environment": {"disturbance_obs": True},
        },
        "weight": 1.0,
    }
    declaration = _track_declaration(
        training_cases=["minimum-phase", shifted],
    )
    track = TrackSpec(declaration)
    with pytest.raises(
        ValueError,
        match="disturbance_obs|observation",
    ):
        track.validate_policy_contract()


def test_one_checkpoint_is_evaluated_on_all_track_cases():
    track = load_track(
        "quadruple-regulation-generalist-v1"
    )
    visited = []

    class FakeEnv:
        def __init__(self, case):
            self.case = case
            self.control_dt = 1.0
            self.episode_steps = 2

        def close(self):
            pass

    def fake_evaluate(controller, env, **kwargs):
        assert controller is checkpoint
        visited.append(env.case.case_id)
        return {
            "metric": "regulation_cost",
            "metric_direction": "minimize",
            "regulation_cost": 2.0,
            "ranking_eligible": True,
        }

    checkpoint = object()
    evaluation = evaluate_policy_on_track(
        checkpoint,
        track,
        split="test",
        base_seeds=[11],
        env_factory=FakeEnv,
        evaluate_fn=fake_evaluate,
    )
    expected = [
        case.case_id for case in track.resolved_cases("test")
    ]
    assert visited == expected
    assert evaluation["case_count"] == len(expected)
    assert len(evaluation["results"]) == len(expected)


def test_specialist_run_is_labeled():
    args = SimpleNamespace(
        name=None,
        algo="sac",
        track=None,
        scenario="quadruple",
        case="minimum-phase",
        policy_scope="specialist",
        seed=3,
    )
    name = run_name_for(args, run_id="fixed")
    assert "specialist" in name
    assert "training-seed3" in name
    assert "regulation" not in name


def test_official_track_rejects_conflicting_cli_overrides():
    args = _training_args(track="quadruple-regulation-generalist-v1")
    args.goal = "economic"
    with pytest.raises(ValueError, match="conflicts with --track"):
        configure_training_track(args)


def test_no_custom_selectors_uses_default_official_track():
    args = _training_args()
    configure_training_track(args)
    assert args.track == "quadruple-regulation-generalist-v1"
    assert args.policy_scope == "generalist"
    assert args.goal == "regulation"
    assert args.reward_spec == "regulation-v1"


def test_track_training_artifact_records_split_provenance():
    track = load_track(
        "quadruple-regulation-generalist-v1"
    )
    training = {
        "track_id": track.id,
        "track_hash": track.track_hash,
        "goal": track.goal,
        "reward_spec_id": track.reward_spec_id,
        "policy_scope": track.policy_scope,
        "training_seed_namespace": track.seed_namespace("training"),
        "validation_seed_namespace": track.seed_namespace("validation"),
    }
    payload = rl_payload(
        kind="test",
        scenario=track.scenario,
        goal=track.goal,
        action_mode=track.policy_contract["action_mode"],
        training=training,
        evaluation=track.metadata(),
        results=[],
        rows=[],
    )
    assert RL_ARTIFACT_SCHEMA_VERSION == "aiogym.rl_training_artifact.v3"
    assert payload["track_id"] == track.id
    assert payload["policy_scope"] == "generalist"
    assert "test_seed_namespace" not in payload
    assert "test_seed_namespace_hash" not in payload


def _track_declaration(
    *,
    scenario="quadruple",
    training_cases=None,
    validation_cases=None,
    test_cases=None,
    control_dt=1.0,
):
    track = load_track(
        "quadruple-regulation-generalist-v1",
        validate_policy_contract=False,
    ).declaration
    track["id"] = "test-track-v1"
    track["scenario"] = scenario
    track["policy_contract"]["control_dt"] = control_dt
    if scenario == "cascade":
        track["policy_contract"].update(
            {
                "disturbance_obs": True,
                "previous_action_obs": False,
                "normalize_observations": False,
                "tracking_error_obs": False,
            }
        )
    track["training"]["cases"] = deepcopy(
        training_cases or ["minimum-phase"]
    )
    track["validation"]["cases"] = deepcopy(
        validation_cases or [track["training"]["cases"][0]]
    )
    track["test"]["cases"] = deepcopy(
        test_cases or [track["training"]["cases"][0]]
    )
    return track


def _training_args(**overrides):
    values = {
        "track": None,
        "scenario": None,
        "case": None,
        "goal": None,
        "reward_spec": None,
        "policy_scope": None,
        "action_mode": None,
        "control_dt": None,
        "disturbance_obs": None,
        "previous_action_obs": None,
        "normalize_observations": None,
        "tracking_error_obs": None,
        "integral_obs": None,
        "steps": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)

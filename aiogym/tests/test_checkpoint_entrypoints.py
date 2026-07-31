"""Phase-4 acceptance tests for unified checkpoint entry points."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from aiogym.cli import benchmark as benchmark_cli
from aiogym.cli.evaluate import main as evaluate_main
from aiogym.cli.final_test import _load_config
from aiogym.controllers.checkpoints import SUPPORTED_POLICY_ALGORITHMS
from aiogym.controllers.checkpoints import (
    LearnedPolicySpec,
    learned_policy_spec_for_track,
)


TRACK_ID = "quadruple-regulation-generalist-v1"


@pytest.mark.parametrize(
    "algorithm_id",
    SUPPORTED_POLICY_ALGORITHMS,
)
def test_evaluate_cli_accepts_every_checkpoint_algorithm(
    tmp_path,
    monkeypatch,
    capsys,
    algorithm_id,
):
    checkpoint = tmp_path / f"{algorithm_id}.checkpoint"
    checkpoint.write_bytes(algorithm_id.encode("ascii"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded = []
    evaluated = []

    def load_policy(spec, **kwargs):
        loaded.append((spec, kwargs))
        return object()

    monkeypatch.setattr(
        "aiogym.cli.evaluate.load_policy_checkpoint",
        load_policy,
    )
    monkeypatch.setattr(
        "aiogym.cli.evaluate.evaluate_policy_on_track",
        lambda *args, **kwargs: evaluated.append(kwargs) or {
            "track_id": TRACK_ID,
            "track_hash": "a" * 64,
            "split": "validation",
            "seed_namespace": "validation",
            "base_seeds": [5000],
            "case_count": 1,
            "aggregate": {"official_score": 1.0},
        },
    )

    assert (
        evaluate_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--track",
                TRACK_ID,
                "--algorithm",
                algorithm_id,
                "--sha256",
                digest,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    spec, kwargs = loaded[0]
    assert spec.algorithm_id == algorithm_id
    assert spec.scenario == "quadruple"
    assert spec.action_mode == "actuator"
    assert spec.observation_dim > 0
    assert spec.action_dim == 2
    assert kwargs["device"] == "cpu"
    assert evaluated
    assert all("split" not in call for call in evaluated)
    assert payload["checkpoint_sha256"] == digest


def test_evaluate_cli_without_digest_pins_current_file(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    digest = hashlib.sha256(b"policy").hexdigest()
    loaded = []
    monkeypatch.setattr(
        "aiogym.cli.evaluate.load_policy_checkpoint",
        lambda spec, **kwargs: loaded.append(spec) or object(),
    )
    monkeypatch.setattr(
        "aiogym.cli.evaluate.evaluate_policy_on_track",
        lambda *args, **kwargs: {
            "track_id": TRACK_ID,
            "track_hash": "a" * 64,
            "split": "validation",
            "seed_namespace": "validation",
            "base_seeds": [5000],
            "case_count": 1,
            "aggregate": {},
        },
    )

    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--track",
            TRACK_ID,
            "--algorithm",
            "bc",
        ]
    )
    assert loaded[0].sha256 == digest
    assert json.loads(capsys.readouterr().out)[
        "checkpoint_sha256"
    ] == digest


def test_evaluate_cli_rejects_wrong_digest_through_unified_loader(
    tmp_path,
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluate_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--track",
                TRACK_ID,
                "--algorithm",
                "bc",
                "--sha256",
                "0" * 64,
            ]
        )


@pytest.mark.parametrize(
    "algorithm_id",
    SUPPORTED_POLICY_ALGORITHMS,
)
def test_final_test_config_accepts_every_checkpoint_algorithm(
    tmp_path,
    algorithm_id,
):
    checkpoint = tmp_path / f"{algorithm_id}.checkpoint"
    checkpoint.write_bytes(algorithm_id.encode("ascii"))
    declaration = {
        "schema_version": "aiogym.final_test.v1",
        "track_id": TRACK_ID,
        "lock_path": str(tmp_path / "lock.json"),
        "base_seeds": [5],
        "checkpoints": {
            algorithm_id: {
                "path": str(checkpoint),
                "algorithm_id": algorithm_id,
                "sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
        },
        "output": str(tmp_path / "final.json"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")

    assert _load_config(path)["checkpoints"][algorithm_id][
        "algorithm_id"
    ] == algorithm_id


def test_benchmark_checkpoint_path_uses_environment_contract(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint = tmp_path / "bc.pt"
    checkpoint.write_bytes(b"bc")
    digest = hashlib.sha256(b"bc").hexdigest()
    output = tmp_path / "benchmark.json"
    loaded = []
    evaluated = []

    def load_policy(spec, **kwargs):
        loaded.append((spec, kwargs))
        return "controller"

    monkeypatch.setattr(
        benchmark_cli,
        "load_policy_checkpoint",
        load_policy,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "make_controller",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark_cli,
        "evaluate_policy_on_track",
        lambda *args, **kwargs: evaluated.append(kwargs) or {
            "split": "validation",
            "aggregate": {
                "official_score": 1.0,
                "metric": "regulation_cost_rate",
                "metric_value": 1.0,
                "ranking_eligible": True,
            },
        },
    )
    assert benchmark_cli.main(
        [
            TRACK_ID,
            "--controllers",
            "pid",
            "--checkpoint",
            str(checkpoint),
            "--algorithm",
            "bc",
            "--sha256",
            digest,
            "--output",
            str(output),
        ]
    ) == 0
    assert capsys.readouterr().out
    spec, kwargs = loaded[0]
    assert isinstance(spec, LearnedPolicySpec)
    assert spec.algorithm_id == "bc"
    assert spec.observation_dim > 0
    assert spec.action_dim == 2
    assert kwargs["device"] == "cpu"
    assert evaluated
    assert all("split" not in call for call in evaluated)
    assert json.loads(output.read_text(encoding="utf-8"))["split"] == (
        "validation"
    )


def test_benchmark_uses_one_canonical_checkpoint_option_family():
    options = {
        option
        for action in benchmark_cli.build_parser()._actions
        for option in action.option_strings
    }
    assert {"--checkpoint", "--algorithm", "--sha256", "--name"} <= options
    assert {
        "--policy-path",
        "--policy-algorithm",
        "--policy-sha256",
        "--sb3-path",
        "--onnx-path",
    }.isdisjoint(options)


def test_track_policy_dimensions_do_not_resolve_locked_test_split(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    visited = []

    class Track:
        def resolved_cases(self, split):
            visited.append(split)
            return (object(),)

    class Env:
        observation_space = SimpleNamespace(shape=(7,))
        action_space = SimpleNamespace(shape=(2,))
        unwrapped = SimpleNamespace(
            scenario="quadruple",
            action_mode="actuator",
        )

        def close(self):
            visited.append("closed")

    monkeypatch.setattr(
        "aiogym._environment.builder.build_track_case_environment",
        lambda track, case, info_level: Env(),
    )
    spec = learned_policy_spec_for_track(
        checkpoint,
        "bc",
        hashlib.sha256(b"policy").hexdigest(),
        Track(),
    )

    assert spec.observation_dim == 7
    assert visited == ["validation", "closed"]

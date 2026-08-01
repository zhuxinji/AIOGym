"""Acceptance tests for verified learned-policy checkpoint loading."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from aiogym.controllers.checkpoints import (
    SUPPORTED_POLICY_ALGORITHMS,
    LearnedPolicySpec,
    load_policy_checkpoint,
)
from aiogym.controllers.contracts import ControllerContext
from aiogym.rl.policy_spec import (
    behavior_cloning_policy_spec,
    rlpd_policy_spec,
)


def test_learned_policy_spec_is_frozen_and_algorithm_set_is_canonical(
    tmp_path,
):
    assert SUPPORTED_POLICY_ALGORITHMS == (
        "sac",
        "td3",
        "ppo",
        "bc",
        "rlpd",
        "onnx",
    )
    spec = LearnedPolicySpec(
        path=tmp_path / "policy.pt",
        algorithm_id="BC",
        sha256="0" * 64,
        scenario="quadruple",
        action_mode="actuator",
        observation_dim=2,
        action_dim=1,
    )
    assert spec.algorithm_id == "bc"
    with pytest.raises(FrozenInstanceError):
        spec.action_dim = 2


def test_hash_mismatch_happens_before_optional_backend_import(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not-an-sb3-checkpoint")
    code = f"""
import sys
from aiogym.controllers.checkpoints import LearnedPolicySpec, load_policy_checkpoint
spec = LearnedPolicySpec(
    path={str(checkpoint)!r},
    algorithm_id="sac",
    sha256={"0" * 64!r},
    scenario="quadruple",
    action_mode="actuator",
    observation_dim=2,
    action_dim=1,
)
try:
    load_policy_checkpoint(spec)
except ValueError as exc:
    assert "SHA256 mismatch" in str(exc)
else:
    raise AssertionError("hash mismatch was accepted")
assert "stable_baselines3" not in sys.modules
assert "torch" not in sys.modules
assert "onnxruntime" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bc_round_trip_scales_normalized_action_exactly_once(tmp_path):
    torch = pytest.importorskip("torch")
    from aiogym.rl.behavior_cloning import BehaviorCloningPolicy

    policy = BehaviorCloningPolicy(2, 1, hidden=4)
    with torch.no_grad():
        for parameter in policy.model.parameters():
            parameter.zero_()
        policy.model[-2].bias.fill_(float(np.arctanh(0.5)))
    checkpoint = tmp_path / "bc.pt"
    torch.save(
        {
            "policy_spec": behavior_cloning_policy_spec(
                2,
                1,
                4,
                scenario="quadruple",
                action_mode="actuator",
            ),
            "policy_state_dict": policy.model.state_dict(),
        },
        checkpoint,
    )

    controller = load_policy_checkpoint(
        _spec(checkpoint, "bc", observation_dim=2, action_dim=1)
    )
    raw, _ = controller.policy.predict(
        np.asarray([0.2, -0.3], dtype=np.float32)
    )
    action = controller.act(
        np.asarray([0.2, -0.3], dtype=np.float32),
        _context(),
    )

    assert raw == pytest.approx([0.5])
    assert action == pytest.approx([0.75])
    assert controller.normalized_actions is True
    assert controller.metadata()["checkpoint"]["sha256"] == _sha256(
        checkpoint
    )


def test_rlpd_round_trip_loads_actor_only(tmp_path):
    torch = pytest.importorskip("torch")
    from aiogym.rl.rlpd import Actor, LOG_STD_MAX, LOG_STD_MIN

    actor = Actor(2, 1, hidden=4)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.mu.bias.fill_(float(np.arctanh(-0.5)))
    checkpoint = tmp_path / "rlpd.pt"
    torch.save(
        {
            "policy_spec": rlpd_policy_spec(
                2,
                1,
                4,
                scenario="quadruple",
                action_mode="actuator",
                log_std_bounds=(LOG_STD_MIN, LOG_STD_MAX),
            ),
            "policy_state_dict": actor.state_dict(),
            "critics": {"must_not": "load"},
            "actor_optimizer": {"must_not": "load"},
            "online_replay": {"must_not": "load"},
        },
        checkpoint,
    )

    controller = load_policy_checkpoint(
        _spec(checkpoint, "rlpd", observation_dim=2, action_dim=1)
    )
    action = controller.act(
        np.asarray([0.2, -0.3], dtype=np.float32),
        _context(),
    )

    assert action == pytest.approx([0.25])
    assert set(vars(controller.policy)) == {
        "device",
        "actor",
        "policy_spec",
        "observation_dim",
        "action_dim",
    }
    assert not hasattr(controller.policy, "online")
    assert not hasattr(controller.policy, "optimizer")


@pytest.mark.parametrize("algorithm_id", ["bc", "rlpd"])
def test_native_checkpoint_without_policy_spec_is_rejected(
    tmp_path,
    algorithm_id,
):
    torch = pytest.importorskip("torch")
    if algorithm_id == "bc":
        from aiogym.rl.behavior_cloning import BehaviorCloningPolicy

        state = BehaviorCloningPolicy(
            2,
            1,
            hidden=4,
        ).model.state_dict()
        payload = {"trainer": {"policy": state}}
    else:
        from aiogym.rl.rlpd import Actor

        state = Actor(2, 1, hidden=4).state_dict()
        payload = {"actor": state}
    checkpoint = tmp_path / f"legacy-{algorithm_id}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(
        ValueError,
        match="requires both policy_spec and policy_state_dict",
    ):
        load_policy_checkpoint(
            _spec(
                checkpoint,
                algorithm_id,
                observation_dim=2,
                action_dim=1,
            )
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"observation_dim": 3}, "observation_dim"),
        ({"action_dim": 2}, "action_dim"),
        ({"scenario": "cstr"}, "scenario"),
        ({"action_mode": "setpoint"}, "action_mode"),
    ],
)
def test_native_policy_spec_mismatch_is_rejected(
    tmp_path,
    override,
    match,
):
    torch = pytest.importorskip("torch")
    from aiogym.rl.behavior_cloning import BehaviorCloningPolicy

    policy = BehaviorCloningPolicy(2, 1, hidden=4)
    checkpoint = tmp_path / "bc.pt"
    torch.save(
        {
            "policy_spec": behavior_cloning_policy_spec(
                2,
                1,
                4,
                scenario="quadruple",
                action_mode="actuator",
            ),
            "policy_state_dict": policy.model.state_dict(),
        },
        checkpoint,
    )
    values = {
        "scenario": "quadruple",
        "action_mode": "actuator",
        "observation_dim": 2,
        "action_dim": 1,
        **override,
    }
    spec = LearnedPolicySpec(
        path=checkpoint,
        algorithm_id="bc",
        sha256=_sha256(checkpoint),
        **values,
    )

    with pytest.raises(ValueError, match=match):
        load_policy_checkpoint(spec)


def test_sb3_dispatch_validates_spaces_and_requests_device(
    tmp_path,
    monkeypatch,
):
    from aiogym.controllers.adapters import SB3PolicyController

    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"fake-sb3")
    calls = {}

    class FakePolicy:
        observation_space = SimpleNamespace(shape=(3,))
        action_space = SimpleNamespace(shape=(2,))

        def predict(self, observation, deterministic=True):
            del observation, deterministic
            return np.zeros(2, dtype=np.float32), None

    def fake_load(cls, path, algo="sac", **kwargs):
        calls.update(path=path, algo=algo, kwargs=kwargs)
        return cls(
            FakePolicy(),
            action_mode=kwargs["action_mode"],
            normalized_actions=kwargs["normalized_actions"],
            control_structure=kwargs["control_structure"],
        )

    monkeypatch.setattr(
        SB3PolicyController,
        "load",
        classmethod(fake_load),
    )
    controller = load_policy_checkpoint(
        _spec(
            checkpoint,
            "td3",
            observation_dim=3,
            action_dim=2,
        ),
        device="cuda:1",
    )

    assert calls["algo"] == "td3"
    assert calls["kwargs"]["device"] == "cuda:1"
    assert controller.normalized_actions is True


@pytest.mark.onnx
def test_onnx_dispatch_passes_complete_shape_contract(
    tmp_path,
    monkeypatch,
):
    from aiogym.controllers.onnx import ONNXPolicyController

    checkpoint = tmp_path / "policy.onnx"
    checkpoint.write_bytes(b"fake-onnx")
    calls = {}

    class FakeONNXController:
        def metadata(self):
            return {"name": "fake-onnx"}

    def fake_load(cls, path, **kwargs):
        calls.update(path=path, kwargs=kwargs)
        return FakeONNXController()

    monkeypatch.setattr(
        ONNXPolicyController,
        "load",
        classmethod(fake_load),
    )
    controller = load_policy_checkpoint(
        _spec(
            checkpoint,
            "onnx",
            observation_dim=3,
            action_dim=2,
        ),
        providers=["CPUExecutionProvider"],
    )

    assert calls["kwargs"]["expected_observation_dim"] == 3
    assert calls["kwargs"]["expected_action_dim"] == 2
    assert calls["kwargs"]["normalized_actions"] is True
    assert controller.metadata()["checkpoint"]["algorithm_id"] == "onnx"


def test_backend_dependency_error_is_actionable_after_hash_passes(
    tmp_path,
    monkeypatch,
):
    from aiogym.controllers.adapters import SB3PolicyController

    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"fake-sb3")

    def unavailable(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "stable-baselines3 is required; install aiogym[rl]"
        )

    monkeypatch.setattr(SB3PolicyController, "load", unavailable)
    with pytest.raises(RuntimeError, match="aiogym\\[rl\\]"):
        load_policy_checkpoint(
            _spec(
                checkpoint,
                "sac",
                observation_dim=3,
                action_dim=2,
            )
        )


def _spec(
    path,
    algorithm_id,
    *,
    observation_dim,
    action_dim,
):
    return LearnedPolicySpec(
        path=path,
        algorithm_id=algorithm_id,
        sha256=_sha256(path),
        scenario="quadruple",
        action_mode="actuator",
        observation_dim=observation_dim,
        action_dim=action_dim,
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context() -> ControllerContext:
    return ControllerContext(
        measurement={},
        setpoint={},
        info={},
        action_mode="actuator",
        control_dt=1.0,
    )

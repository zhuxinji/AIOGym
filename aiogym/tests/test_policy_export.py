from __future__ import annotations

import hashlib

import numpy as np
import pytest

from aiogym.controllers.checkpoints import (
    LearnedPolicySpec,
    checkpoint_sha256,
)
from aiogym.controllers.export import export_policy_checkpoint
from aiogym.rl.policy_spec import behavior_cloning_policy_spec


pytestmark = [pytest.mark.rl, pytest.mark.onnx]


def _bc_checkpoint(tmp_path):
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
    spec = LearnedPolicySpec(
        path=checkpoint,
        algorithm_id="bc",
        sha256=checkpoint_sha256(checkpoint),
        scenario="quadruple",
        action_mode="actuator",
        observation_dim=2,
        action_dim=1,
    )
    return checkpoint, spec


def test_export_status_hash_and_onnx_runtime_output(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    checkpoint, spec = _bc_checkpoint(tmp_path)
    target = tmp_path / "policy.onnx"
    result = export_policy_checkpoint(spec, output=target)
    assert result.status == "success"
    assert result.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    session = ort.InferenceSession(str(target))
    output = session.run(
        None, {"obs": np.asarray([[0.1, -0.2]], dtype=np.float32)}
    )[0]
    assert output.shape == (1, 1)
    assert checkpoint.is_file()


def test_native_checkpoint_survives_optional_export_failure(
    tmp_path, monkeypatch
):
    checkpoint, spec = _bc_checkpoint(tmp_path)
    original = checkpoint.read_bytes()
    monkeypatch.setattr(
        "aiogym.controllers.export.load_policy_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected export failure")
        ),
    )
    result = export_policy_checkpoint(
        spec, output=tmp_path / "failed.onnx"
    )
    assert result.status == "failed"
    assert "injected export failure" in result.error
    assert checkpoint.read_bytes() == original
    assert not (tmp_path / "failed.onnx").exists()

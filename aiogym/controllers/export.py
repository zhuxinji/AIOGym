"""Best-effort exports from one verified canonical policy checkpoint."""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkpoints import LearnedPolicySpec, load_policy_checkpoint


@dataclass(frozen=True)
class ExportResult:
    format: str
    status: str
    path: str | None = None
    sha256: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def export_policy_checkpoint(
    policy_spec: LearnedPolicySpec,
    *,
    format: str = "onnx",
    output: str | Path,
    device: str = "cpu",
) -> ExportResult:
    """Export a verified policy without changing native checkpoint success."""

    if not isinstance(policy_spec, LearnedPolicySpec):
        raise TypeError("policy_spec must be a LearnedPolicySpec")
    if format != "onnx":
        raise ValueError("only ONNX export is supported")
    target = Path(output)
    try:
        import torch

        controller = load_policy_checkpoint(policy_spec, device=device)
        module = _export_module(controller.policy, policy_spec.algorithm_id)
        module.eval()
        dummy = torch.zeros(
            1,
            policy_spec.observation_dim,
            device=_module_device(module),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        options = {
            "input_names": ["obs"],
            "output_names": ["action"],
            "opset_version": 17,
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            options["dynamo"] = False
        torch.onnx.export(module, dummy, str(target), **options)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return ExportResult(
            format="onnx",
            status="success",
            path=str(target),
            sha256=digest,
        )
    except BaseException as exc:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        return ExportResult(
            format="onnx",
            status="failed",
            path=str(target),
            error=f"{type(exc).__name__}: {exc}",
        )


def _export_module(policy, algorithm_id: str):
    import torch

    if algorithm_id in {"ppo", "sac", "td3"}:
        sb3_policy = policy.policy

        class SB3Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = sb3_policy

            def forward(self, obs):
                return self.policy._predict(obs, deterministic=True)

        return SB3Module()
    if algorithm_id == "bc":
        return policy.model
    if algorithm_id == "rlpd":
        actor = policy.actor

        class RLPDModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                mean, _ = self.actor(obs)
                return torch.tanh(mean)

        return RLPDModule()
    raise ValueError(f"unsupported export algorithm: {algorithm_id}")


def _module_device(module):
    import torch

    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


__all__ = ["ExportResult", "export_policy_checkpoint"]

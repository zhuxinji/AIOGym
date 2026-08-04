"""Verified, lazy loading for learned-policy checkpoints."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogym._internal.serialization import file_sha256


SUPPORTED_POLICY_ALGORITHMS = (
    "sac",
    "td3",
    "ppo",
    "bc",
    "rlpd",
    "onnx",
)
SUPPORTED_LEARNED_POLICY_ALGORITHMS = SUPPORTED_POLICY_ALGORITHMS
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LearnedPolicySpec:
    """Complete identity and shape contract for one learned checkpoint."""

    path: str | Path
    algorithm_id: str
    sha256: str
    scenario: str
    action_mode: str
    observation_dim: int
    action_dim: int
    reward_spec_id: str | None = None
    reward_spec_hash: str | None = None

    def __post_init__(self) -> None:
        path = Path(self.path)
        algorithm_id = str(self.algorithm_id).lower()
        if algorithm_id not in SUPPORTED_POLICY_ALGORITHMS:
            raise ValueError(
                "unsupported learned-policy algorithm_id; expected one of: "
                + ", ".join(SUPPORTED_POLICY_ALGORITHMS)
            )
        if not _SHA256_PATTERN.fullmatch(str(self.sha256)):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ValueError("scenario must be a non-empty string")
        if self.action_mode not in {"actuator", "setpoint"}:
            raise ValueError("action_mode must be actuator or setpoint")
        for name in ("observation_dim", "action_dim"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (self.reward_spec_id is None) != (self.reward_spec_hash is None):
            raise ValueError(
                "reward_spec_id and reward_spec_hash must be provided together"
            )
        if self.reward_spec_id is not None:
            from aiogym.rewards.registry import get_reward_spec

            canonical = get_reward_spec(self.reward_spec_id)
            if self.reward_spec_hash != canonical.spec_hash:
                raise ValueError("reward_spec_hash does not match reward_spec_id")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "algorithm_id", algorithm_id)


def learned_policy_spec_for_environment(
    path: str | Path,
    algorithm_id: str,
    sha256: str,
    env,
) -> LearnedPolicySpec:
    """Bind a checkpoint declaration to a constructed policy environment."""

    observation_dim = _vector_space_dimension(
        getattr(env, "observation_space", None),
        "observation",
    )
    action_dim = _vector_space_dimension(
        getattr(env, "action_space", None),
        "action",
    )
    unwrapped = getattr(env, "unwrapped", env)
    scenario = getattr(unwrapped, "scenario", None)
    action_mode = getattr(unwrapped, "action_mode", None)
    return LearnedPolicySpec(
        path=path,
        algorithm_id=algorithm_id,
        sha256=sha256,
        scenario=scenario,
        action_mode=action_mode,
        observation_dim=observation_dim,
        action_dim=action_dim,
        reward_spec_id=getattr(unwrapped, "reward_spec_id", None),
        reward_spec_hash=getattr(unwrapped, "reward_spec_hash", None),
    )


def learned_policy_spec_for_track(
    path: str | Path,
    algorithm_id: str,
    sha256: str,
    track,
) -> LearnedPolicySpec:
    """Resolve dimensions without consulting a Track's locked test split."""

    from aiogym._environment.builder import (
        build_track_case_environment,
    )

    case = track.resolved_cases("validation")[0]
    env = build_track_case_environment(track, case, info_level="minimal")
    try:
        return learned_policy_spec_for_environment(
            path,
            algorithm_id,
            sha256,
            env,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def checkpoint_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 identity of one checkpoint file."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return file_sha256(checkpoint)


def load_policy_checkpoint(
    checkpoint: LearnedPolicySpec | str | Path,
    *,
    algorithm_id: str | None = None,
    sha256: str | None = None,
    expected_sha256: str | None = None,
    scenario: str | None = None,
    action_mode: str | None = None,
    observation_dim: int | None = None,
    action_dim: int | None = None,
    device: str = "cpu",
    providers=None,
):
    """Load a verified checkpoint as an environment-facing controller.

    Optional backends are imported only after the checkpoint hash matches.
    Passing a path directly is a convenience equivalent to constructing a
    :class:`LearnedPolicySpec`.
    """

    spec = _coerce_spec(
        checkpoint,
        algorithm_id=algorithm_id,
        sha256=sha256,
        expected_sha256=expected_sha256,
        scenario=scenario,
        action_mode=action_mode,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    if not spec.path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {spec.path}")
    actual_sha256 = file_sha256(spec.path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"checkpoint SHA256 mismatch for {spec.path}: "
            f"expected {spec.sha256}, got {actual_sha256}"
        )

    if spec.algorithm_id in {"sac", "td3", "ppo"}:
        controller = _load_sb3(spec, device=device)
    elif spec.algorithm_id == "onnx":
        controller = _load_onnx(spec, providers=providers)
    else:
        controller = _load_native_torch(spec, device=device)
    _attach_checkpoint_metadata(controller, spec, actual_sha256)
    return controller


def _coerce_spec(
    checkpoint,
    *,
    algorithm_id,
    sha256,
    expected_sha256,
    scenario,
    action_mode,
    observation_dim,
    action_dim,
) -> LearnedPolicySpec:
    if isinstance(checkpoint, LearnedPolicySpec):
        duplicates = {
            "algorithm_id": algorithm_id,
            "sha256": sha256,
            "expected_sha256": expected_sha256,
            "scenario": scenario,
            "action_mode": action_mode,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        }
        if any(value is not None for value in duplicates.values()):
            raise TypeError(
                "checkpoint attributes must not be repeated with "
                "LearnedPolicySpec"
            )
        return checkpoint
    digest = sha256 if sha256 is not None else expected_sha256
    if sha256 is not None and expected_sha256 is not None:
        raise TypeError("pass only one of sha256 or expected_sha256")
    missing = [
        name
        for name, value in (
            ("algorithm_id", algorithm_id),
            ("sha256", digest),
            ("scenario", scenario),
            ("action_mode", action_mode),
            ("observation_dim", observation_dim),
            ("action_dim", action_dim),
        )
        if value is None
    ]
    if missing:
        raise TypeError(
            "loading a checkpoint path requires: " + ", ".join(missing)
        )
    return LearnedPolicySpec(
        path=checkpoint,
        algorithm_id=algorithm_id,
        sha256=digest,
        scenario=scenario,
        action_mode=action_mode,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )


def _load_sb3(spec: LearnedPolicySpec, *, device: str):
    from .adapters import SB3PolicyController

    controller = SB3PolicyController.load(
        str(spec.path),
        algo=spec.algorithm_id,
        device=device,
        action_mode=spec.action_mode,
        normalized_actions=True,
        control_structure="sb3_policy",
    )
    policy = controller.policy
    _validate_space_dimension(
        getattr(policy, "observation_space", None),
        spec.observation_dim,
        "observation",
    )
    _validate_space_dimension(
        getattr(policy, "action_space", None),
        spec.action_dim,
        "action",
    )
    return controller


def _load_onnx(spec: LearnedPolicySpec, *, providers):
    from .onnx import ONNXPolicyController

    controller = ONNXPolicyController.load(
        str(spec.path),
        providers=providers,
        action_mode=spec.action_mode,
        expected_observation_dim=spec.observation_dim,
        expected_action_dim=spec.action_dim,
        scenario=spec.scenario,
        normalized_actions=True,
    )
    return controller


def _load_native_torch(spec: LearnedPolicySpec, *, device: str):
    payload = _torch_load(spec.path)
    policy_spec, policy_state_dict = _native_checkpoint_components(
        payload,
        spec,
    )
    if spec.algorithm_id == "bc":
        from .native_policies import NativeBehaviorCloningPolicy

        policy = NativeBehaviorCloningPolicy(
            policy_spec,
            policy_state_dict,
            device=device,
        )
    else:
        from .native_policies import NativeRLPDPolicy

        policy = NativeRLPDPolicy(
            policy_spec,
            policy_state_dict,
            device=device,
        )
    from .adapters import PolicyController

    return PolicyController(
        policy,
        name=policy.name,
        action_mode=spec.action_mode,
        control_structure=f"{spec.algorithm_id}_policy",
        normalized_actions=True,
    )


def _torch_load(path: Path):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required for native BC/RLPD policies; install "
            "AIO-Gym with `pip install 'aiogym[rl]'`"
        ) from exc
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("native policy checkpoint must contain a mapping")
    return payload


def _native_checkpoint_components(
    payload: Mapping[str, Any],
    requested: LearnedPolicySpec,
) -> tuple[dict, Mapping[str, Any]]:
    from aiogym.rl.policy_spec import validate_policy_spec

    embedded = payload.get("policy_spec")
    state = payload.get("policy_state_dict")
    if embedded is None or state is None:
        raise ValueError(
            "native checkpoint requires both policy_spec and "
            "policy_state_dict"
        )
    policy_spec = validate_policy_spec(embedded)
    if not isinstance(state, Mapping):
        raise TypeError("native policy_state_dict must be a mapping")
    _validate_embedded_policy_spec(policy_spec, requested)
    return policy_spec, state

def _validate_embedded_policy_spec(
    embedded: Mapping[str, Any],
    requested: LearnedPolicySpec,
) -> None:
    expected = {
        "algorithm_id": requested.algorithm_id,
        "action_mode": requested.action_mode,
        "observation_dim": requested.observation_dim,
        "action_dim": requested.action_dim,
    }
    mismatches = [
        name
        for name, value in expected.items()
        if embedded[name] != value
    ]
    embedded_scenario = embedded["scenario"]
    if (
        embedded_scenario is not None
        and embedded_scenario != requested.scenario
    ):
        mismatches.append("scenario")
    if mismatches:
        raise ValueError(
            "checkpoint policy_spec does not match requested "
            + ", ".join(mismatches)
        )


def _validate_space_dimension(space, expected: int, kind: str) -> None:
    actual = _vector_space_dimension(space, kind, owner="SB3 checkpoint")
    if actual != expected:
        raise ValueError(
            f"SB3 checkpoint {kind} dimension is {actual}, expected {expected}"
        )


def _vector_space_dimension(
    space,
    kind: str,
    *,
    owner: str = "policy environment",
) -> int:
    shape = getattr(space, "shape", None)
    if (
        shape is None
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        raise ValueError(
            f"{owner} has no one-dimensional {kind} space"
        )
    return int(shape[0])


def _attach_checkpoint_metadata(
    controller,
    spec: LearnedPolicySpec,
    actual_sha256: str,
) -> None:
    controller.checkpoint_metadata = {
        "path": str(spec.path),
        "algorithm_id": spec.algorithm_id,
        "sha256": actual_sha256,
        "scenario": spec.scenario,
        "action_mode": spec.action_mode,
        "observation_dim": spec.observation_dim,
        "action_dim": spec.action_dim,
        "reward_spec_id": spec.reward_spec_id,
        "reward_spec_hash": spec.reward_spec_hash,
    }
    original_metadata = controller.metadata

    def metadata():
        return {
            **original_metadata(),
            "checkpoint": dict(controller.checkpoint_metadata),
        }

    controller.metadata = metadata


__all__ = [
    "SUPPORTED_LEARNED_POLICY_ALGORITHMS",
    "SUPPORTED_POLICY_ALGORITHMS",
    "LearnedPolicySpec",
    "checkpoint_sha256",
    "learned_policy_spec_for_environment",
    "learned_policy_spec_for_track",
    "load_policy_checkpoint",
]

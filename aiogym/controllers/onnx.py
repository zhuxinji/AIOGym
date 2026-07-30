"""ONNX Runtime policy controller for exported AIO-Gym policies."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import CONTROLLER_API_VERSION, ControllerContext


class ONNXPolicyController:
    """Run a local obs-to-action ONNX policy through the controller contract."""

    controller_api_version = CONTROLLER_API_VERSION

    def __init__(
        self,
        session,
        *,
        path: str,
        name: str = "ONNX-policy",
        action_mode: str = "setpoint",
        control_structure: str = "onnx_policy",
        input_name: str | None = None,
        output_name: str | None = None,
        expected_action_dim: int | None = None,
        scenario: str | None = None,
        normalized_actions: bool = False,
    ):
        if action_mode not in {"actuator", "setpoint"}:
            raise ValueError("action_mode must be one of: actuator, setpoint")
        self.session = session
        self.path = str(path)
        self.name = name
        self.action_mode = action_mode
        self.control_structure = control_structure
        self.scenario = scenario
        self.normalized_actions = bool(normalized_actions)
        self.input_name, self.input_shape = _resolve_node(
            session.get_inputs(), input_name, "input"
        )
        self.output_name, self.output_shape = _resolve_node(
            session.get_outputs(), output_name, "output"
        )
        self.expected_obs_dim = _fixed_last_dim(self.input_shape)
        self.expected_action_dim = expected_action_dim
        output_dim = _fixed_last_dim(self.output_shape)
        if expected_action_dim is not None and output_dim is not None and output_dim != expected_action_dim:
            raise ValueError(
                f"ONNX policy outputs {output_dim} actions, expected {expected_action_dim}"
            )

    @classmethod
    def load(cls, path: str, *, providers=None, **kwargs):
        policy_path = Path(path)
        if not policy_path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                "onnxruntime is required for ONNX policies; install AIO-Gym "
                "with `pip install 'aiogym[onnx]'`"
            ) from ex
        session_kwargs = {"providers": list(providers)} if providers else {}
        session = ort.InferenceSession(str(policy_path), **session_kwargs)
        return cls(session, path=str(policy_path), **kwargs)

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, context: ControllerContext) -> np.ndarray:
        values = np.asarray(obs, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("ONNX policy observation values must be finite")
        if self.expected_obs_dim is not None and values.size != self.expected_obs_dim:
            raise ValueError(
                f"ONNX policy expects {self.expected_obs_dim} observations, got {values.size}"
            )
        output = self.session.run(
            [self.output_name],
            {self.input_name: values.reshape(1, -1)},
        )[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        if self.expected_action_dim is not None and action.size != self.expected_action_dim:
            raise ValueError(
                f"ONNX policy produced {action.size} actions, expected {self.expected_action_dim}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("ONNX policy produced a non-finite action")
        if self.normalized_actions:
            action = np.clip(0.5 * (action + 1.0), 0.0, 1.0)
        return action

    def metadata(self):
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "api": self.controller_api_version,
            "scenario": self.scenario,
            "path": self.path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "action_mode": self.action_mode,
            "normalized_actions": self.normalized_actions,
            "control_structure": self.control_structure,
        }


def _resolve_node(nodes, requested: str | None, kind: str):
    values = list(nodes)
    if not values:
        raise ValueError(f"ONNX policy must expose at least one {kind}")
    if requested is None:
        node = values[0]
    else:
        node = next((value for value in values if value.name == requested), None)
        if node is None:
            available = ", ".join(value.name for value in values)
            raise ValueError(f"unknown ONNX {kind} {requested!r}; available: {available}")
    return node.name, tuple(node.shape or ())


def _fixed_last_dim(shape):
    if not shape:
        return None
    value = shape[-1]
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

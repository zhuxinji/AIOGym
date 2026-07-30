"""Versioned safety-gate bindings owned by benchmark Tracks."""
from __future__ import annotations

from types import MappingProxyType

from aiogym.evaluation.safety_gate import SafetyGateSpec


_SAFETY_GATES = MappingProxyType(
    {
        "zero-hard-failure-v1": SafetyGateSpec(
            mode="ordinary",
            max_state_violation_count=0.0,
            max_command_violation_count=0.0,
        ),
        "recovery-zero-created-hard-failure-v1": SafetyGateSpec(
            mode="recovery",
            max_state_violation_count=0.0,
            max_command_violation_count=0.0,
        ),
    }
)


def list_safety_gates() -> tuple[str, ...]:
    return tuple(sorted(_SAFETY_GATES))


def get_safety_gate_spec(gate_id: str) -> SafetyGateSpec:
    try:
        return _SAFETY_GATES[str(gate_id)]
    except KeyError as exc:
        available = ", ".join(list_safety_gates())
        raise ValueError(
            f"unknown benchmark safety gate {gate_id!r}; "
            f"available gates: {available}"
        ) from exc


SAFETY_GATES = _SAFETY_GATES

__all__ = ["SAFETY_GATES", "get_safety_gate_spec", "list_safety_gates"]

"""Public-to-physical actuator layouts for parameterized three-tank rigs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorLayout:
    public_names: tuple[str, ...]
    internal_names: tuple[str, ...]
    public_to_internal: tuple[int, ...]

    def __post_init__(self):
        if len(self.public_names) != len(self.public_to_internal):
            raise ValueError("public actuator names and mapping must have equal length")
        if len(set(self.public_names)) != len(self.public_names):
            raise ValueError("public actuator names must be unique")
        if len(set(self.internal_names)) != len(self.internal_names):
            raise ValueError("internal actuator names must be unique")
        if len(set(self.public_to_internal)) != len(self.public_to_internal):
            raise ValueError("public actuator mapping must be one-to-one")
        if any(
            index < 0 or index >= len(self.internal_names)
            for index in self.public_to_internal
        ):
            raise ValueError("public actuator mapping contains an invalid internal index")
        for name, index in zip(self.public_names, self.public_to_internal):
            if name != self.internal_names[index]:
                raise ValueError("public actuator names must match their internal slots")

    def expand(self, action):
        values = _vector_items(action, len(self.public_names), "public")
        zero = values[0] * 0 if values else 0.0
        internal = [zero for _ in self.internal_names]
        for value, index in zip(values, self.public_to_internal):
            internal[index] = value
        return internal

    def compress(self, internal_action):
        values = _vector_items(
            internal_action,
            len(self.internal_names),
            "internal",
        )
        return [values[index] for index in self.public_to_internal]


def _vector_items(value, expected, label):
    try:
        length = int(value.shape[0]) if hasattr(value, "shape") else len(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} action must be a flat vector") from error
    if length != expected:
        raise ValueError(
            f"{label} action must contain {expected} values, got {length}"
        )
    return [value[index] for index in range(length)]


__all__ = ["ActuatorLayout"]

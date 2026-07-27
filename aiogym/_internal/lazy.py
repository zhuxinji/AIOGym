"""Small helpers for package-level lazy public APIs."""
from __future__ import annotations

from importlib import import_module
from typing import Mapping


def resolve_export(
    namespace: dict,
    package: str,
    exports: Mapping[str, str],
    name: str,
):
    """Import, cache, and return one declared package export."""

    module_name = exports.get(name)
    if module_name is None:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, package), name)
    namespace[name] = value
    return value


def exported_dir(namespace: Mapping, exports: Mapping[str, str]) -> list[str]:
    """Return module globals plus declared lazy exports for introspection."""

    return sorted(set(namespace) | set(exports))


__all__ = ["exported_dir", "resolve_export"]

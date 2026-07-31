"""Unified controller public API with optional implementations loaded lazily."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "Controller": ".contracts",
    "make_controller": ".registry",
    "register_controller": ".registry",
    "unregister_controller": ".registry",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)

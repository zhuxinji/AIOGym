"""Process-model public API with implementation groups loaded lazily."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "ProcessModelContract": ".core",
    "define_model": ".declarative",
    "make_model": ".registry",
    "register_model": ".registry",
    "unregister_model": ".registry",
    "CaseSpec": ".cases",
    "list_cases": ".cases",
    "load_case": ".cases",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)

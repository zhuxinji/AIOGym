"""Stable Dataset v2 collection, reading, and validation facade."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "DatasetReader": ".reader",
    "validate_dataset": ".reader",
    "list_collectors": ".collector",
    "DatasetCollectionConfig": ".config",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)

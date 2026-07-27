"""Path resolution shared by artifact readers and validators."""
from __future__ import annotations

from pathlib import Path


def resolve_artifact_path(root: Path, raw, default: str) -> Path:
    """Resolve an artifact manifest value against its artifact directory."""

    if raw:
        path = Path(raw)
        if path.is_absolute() or path.exists():
            return path
        return root / path
    return root / default

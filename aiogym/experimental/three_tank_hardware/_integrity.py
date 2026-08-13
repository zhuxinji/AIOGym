"""Content digest used only by experimental hardware records."""
from __future__ import annotations

import hashlib

from aiogym.core.io import canonical_json_bytes


def content_digest(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = ["content_digest"]

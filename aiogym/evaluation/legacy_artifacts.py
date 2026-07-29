"""Explicit, offline migration reader for pre-redesign evaluation artifacts.

This module is intentionally outside the live environment and evaluation
pipeline.  It exists only so archived experiment files can be inspected or
converted by a deliberate migration step.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def load_legacy_evaluation_artifact(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Read an archived v4/v5 artifact into the current field vocabulary."""

    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        with Path(source).open(encoding="utf-8") as stream:
            payload = json.load(stream)
    return migrate_legacy_evaluation_artifact(payload)


def migrate_legacy_evaluation_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert nested archived results and rows without changing live APIs."""

    migrated = deepcopy(dict(payload))
    if "rows" in migrated:
        migrated["rows"] = [
            _migrate_record(row) for row in migrated.get("rows", [])
        ]
    if "results" in migrated:
        migrated["results"] = [
            _migrate_record(result) for result in migrated.get("results", [])
        ]
    if isinstance(migrated.get("result"), Mapping):
        migrated["result"] = _migrate_record(migrated["result"])
    if "controller_name" in migrated:
        migrated = _migrate_record(migrated)
    return migrated


def _migrate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(dict(record))
    old_goal = str(migrated.pop("objective", "") or "")
    migrated.setdefault(
        "goal",
        "economic" if old_goal == "economic" else "regulation",
    )
    if "task" in migrated and "case" not in migrated:
        migrated["case"] = migrated.pop("task")
    if "suite_case" in migrated and "run_case_id" not in migrated:
        migrated["run_case_id"] = migrated.pop("suite_case")
    if "objective_status" in migrated and "acceptance_status" not in migrated:
        migrated["acceptance_status"] = migrated.pop("objective_status")
    if "objective_acceptance" in migrated and "acceptance" not in migrated:
        migrated["acceptance"] = migrated.pop("objective_acceptance")
    return migrated

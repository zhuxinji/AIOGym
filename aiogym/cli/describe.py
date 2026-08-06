"""Deterministic resource catalogs and human-readable descriptions."""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from aiogym.benchmarks.tracks.registry import (
    iter_track_catalog,
    load_track,
    preferred_track_selector,
)
from aiogym.rewards.registry import (
    get_reward_spec,
    iter_reward_catalog,
    resolve_reward_id,
    reward_display,
)


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Format a stable, dependency-free table without terminal assumptions."""

    rendered = [tuple(str(value) for value in row) for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        if len(row) != len(headers):
            raise ValueError("table rows must have the same width as headers")
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = [
        "  ".join(
            header.ljust(width) for header, width in zip(headers, widths)
        ).rstrip()
    ]
    lines.extend(
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
        for row in rendered
    )
    return "\n".join(lines)


def track_catalog(*, include_all: bool = False) -> tuple[dict[str, Any], ...]:
    return iter_track_catalog(include_all=include_all)


def reward_catalog() -> tuple[dict[str, Any], ...]:
    return iter_reward_catalog()


def profile_catalog(
    *,
    kind: str | None = None,
    target: str | None = None,
    algorithm: str | None = None,
) -> tuple[dict[str, Any], ...]:
    from aiogym.datasets.profiles import list_collection_profiles
    from aiogym.rl.profiles import list_training_profiles

    rows = []
    if kind in {None, "training"}:
        for profile in list_training_profiles(
            track_id=target,
            algorithm_id=algorithm,
        ):
            rows.append(
                {
                    "kind": "training",
                    "profile": profile.selector,
                    "target": (
                        preferred_track_selector(profile.track_id)
                        or profile.track_id
                    ),
                    "algorithm": profile.algorithm_id,
                    "profile_id": profile.id,
                    "track_id": profile.track_id,
                    "description": profile.description,
                    "source": profile.source,
                }
            )
    if kind in {None, "collection"} and algorithm is None:
        for profile in list_collection_profiles(track_id=target):
            rows.append(
                {
                    "kind": "collection",
                    "profile": profile.selector,
                    "target": (
                        preferred_track_selector(profile.track_id)
                        or profile.track_id
                    ),
                    "algorithm": "-",
                    "profile_id": profile.id,
                    "track_id": profile.track_id,
                    "description": profile.description,
                    "source": profile.source,
                    "default_transitions": profile.default_transitions,
                    "default_workers": profile.default_workers,
                    "collectors": [dict(item) for item in profile.collectors],
                }
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row["kind"],
                row["target"],
                row["algorithm"],
                row["profile_id"],
            ),
        )
    )


def render_track_catalog(
    *,
    ids: bool = False,
    include_all: bool = False,
    json_output: bool = False,
) -> str:
    if ids:
        from aiogym.benchmarks.tracks.registry import list_tracks

        return "\n".join(list_tracks())
    rows = track_catalog(include_all=include_all)
    if json_output:
        return json.dumps(rows, indent=2, sort_keys=True)
    return format_table(
        ("SELECTOR", "GOAL", "SCOPE", "CANONICAL ID"),
        (
            (
                row["selector"],
                row["goal"],
                row["policy_scope"],
                row["track_id"],
            )
            for row in rows
        ),
    )


def render_reward_catalog(*, json_output: bool = False) -> str:
    rows = reward_catalog()
    if json_output:
        return json.dumps(rows, indent=2, sort_keys=True)
    return format_table(
        ("SELECTOR", "CANONICAL ID", "GOAL", "SUMMARY"),
        (
            (
                row["selector"],
                row["reward_spec_id"],
                row["goal"],
                row["summary"],
            )
            for row in rows
        ),
    )


def render_profile_catalog(
    *,
    kind: str | None = None,
    target: str | None = None,
    algorithm: str | None = None,
    json_output: bool = False,
) -> str:
    rows = profile_catalog(kind=kind, target=target, algorithm=algorithm)
    if json_output:
        return json.dumps(rows, indent=2, sort_keys=True)
    return format_table(
        (
            "KIND",
            "PROFILE",
            "TARGET",
            "ALGORITHM",
            "CANONICAL PROFILE ID",
        ),
        (
            (
                row["kind"],
                row["profile"],
                row["target"],
                row["algorithm"],
                row["profile_id"],
            )
            for row in rows
        ),
    )


def describe_track(value: str) -> dict[str, Any]:
    track = load_track(value, validate_policy_contract=False)
    return {
        "selector": preferred_track_selector(track.id),
        "track_id": track.id,
        "track_hash": track.track_hash,
        "description": str(track.declaration.get("description", "")),
        "scenario": track.scenario,
        "goal": track.goal,
        "policy_scope": track.policy_scope,
        "reward_spec_id": track.reward_spec_id,
        "scorecard_spec_id": track.scorecard_spec_id,
        "ranking_spec_id": track.ranking_spec_id,
        "policy_contract": track.policy_contract,
    }


def render_track_description(value: str, *, json_output: bool = False) -> str:
    row = describe_track(value)
    if json_output:
        return json.dumps(row, indent=2, sort_keys=True)
    contract = row["policy_contract"]
    fields = (
        ("Selector", row["selector"] or "-"),
        ("Canonical ID", row["track_id"]),
        ("Scenario", row["scenario"]),
        ("Goal", row["goal"]),
        ("Scope", row["policy_scope"]),
        ("RewardSpec", row["reward_spec_id"]),
        ("Scorecard", row["scorecard_spec_id"]),
        ("Ranking", row["ranking_spec_id"]),
        ("Track hash", row["track_hash"]),
        ("Description", row["description"]),
    )
    return _label_block(fields) + "\n\nPolicy contract\n" + format_table(
        ("FIELD", "VALUE"),
        (
            (key, json.dumps(value, sort_keys=True))
            for key, value in sorted(contract.items())
        ),
    )


def describe_reward(value: str) -> dict[str, Any]:
    reward_id = resolve_reward_id(value)
    spec = get_reward_spec(reward_id)
    display = reward_display(reward_id)
    used_by = tuple(
        row["selector"]
        for row in track_catalog()
        if row["reward_spec_id"] == reward_id
    )
    return {
        **display,
        "goal": spec.goal,
        "discretization": spec.metadata.get("discretization"),
        "term_weights": dict(spec.term_weights),
        "cost_weights": dict(spec.cost_weights),
        "terminal_failure_cost_rate": spec.terminal_failure_cost_rate,
        "used_by": used_by,
        "reward_spec_hash": spec.spec_hash,
    }


def render_reward_description(value: str, *, json_output: bool = False) -> str:
    row = describe_reward(value)
    if json_output:
        return json.dumps(row, indent=2, sort_keys=True)
    fields = (
        ("Selector", row["selector"]),
        ("Canonical ID", row["reward_spec_id"]),
        ("Goal", row["goal"]),
        ("Discretization", row["discretization"]),
        ("Summary", row["summary"]),
        ("RewardSpec hash", row["reward_spec_hash"]),
    )
    terms = format_table(
        ("TERM", "WEIGHT", "DESCRIPTION"),
        (
            (
                name,
                weight,
                row["term_descriptions"].get(name, ""),
            )
            for name, weight in row["term_weights"].items()
        ),
    )
    costs = format_table(
        ("CHANNEL", "WEIGHT", "DESCRIPTION"),
        (
            (
                name,
                weight,
                row["cost_descriptions"].get(name, ""),
            )
            for name, weight in row["cost_weights"].items()
        ),
    )
    return (
        _label_block(fields)
        + "\n\nReward terms\n"
        + terms
        + "\n\nCost channels\n"
        + costs
        + "\n\nTerminal failure cost rate: "
        + str(row["terminal_failure_cost_rate"])
        + "\nUsed by: "
        + ", ".join(row["used_by"])
    )


def render_profile_description(
    value: str,
    *,
    target: str | None = None,
    algorithm: str | None = None,
    json_output: bool = False,
) -> str:
    matches = tuple(
        row
        for row in profile_catalog(target=target, algorithm=algorithm)
        if value in {row["profile"], row["profile_id"]}
    )
    if not matches:
        raise ValueError(f"no profile matches {value!r} and the supplied filters")
    if json_output:
        payload: Any = matches[0] if len(matches) == 1 else matches
        return json.dumps(payload, indent=2, sort_keys=True)
    if len(matches) > 1:
        return render_profile_catalog(
            target=target,
            algorithm=algorithm,
            json_output=False,
        )
    row = matches[0]
    fields = (
        ("Kind", row["kind"]),
        ("Profile", row["profile"]),
        ("Canonical profile ID", row["profile_id"]),
        ("Target", row["target"]),
        ("Canonical Track", row["track_id"]),
        ("Algorithm", row["algorithm"]),
        ("Description", row["description"]),
        ("Source", row["source"]),
    )
    return _label_block(fields)


def _label_block(fields: Iterable[tuple[str, Any]]) -> str:
    rows = tuple((str(label), str(value)) for label, value in fields)
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        f"{label + ':':{width + 1}s} {value}" for label, value in rows
    )


__all__ = [
    "describe_reward",
    "describe_track",
    "format_table",
    "profile_catalog",
    "render_profile_catalog",
    "render_profile_description",
    "render_reward_catalog",
    "render_reward_description",
    "render_track_catalog",
    "render_track_description",
]

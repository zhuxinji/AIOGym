"""Dependency-free rendering for reusable training-curve artifacts."""
from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._svg import map_value, plot_range


TRAINING_CURVE_SCHEMA_VERSION = "aiogym.training_curve.v2"


def plot_training_curve(
    curve: Mapping[str, Any] | str | Path,
    *,
    output: str | Path,
) -> Path:
    """Render one training-curve mapping or JSON artifact as SVG."""

    payload = _load_curve(curve)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_training_curve_svg(payload), encoding="utf-8")
    return target


def _load_curve(curve: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(curve, Mapping):
        return dict(curve)
    path = Path(curve)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read training curve {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("training curve JSON must contain an object")
    return payload


def _training_curve_svg(curve: Mapping[str, Any]) -> str:
    _validate_curve(curve)
    records = list(curve["records"])
    episodes = list(curve["episodes"])
    total_steps = int(curve["actual_steps"])
    record_every = int(curve["record_every"])
    panels = (
        {
            "title": f"Mean step reward ({record_every}-step windows)",
            "x": [row["end_step"] for row in records],
            "y": [row["mean_reward"] for row in records],
            "outcomes": None,
            "color": "#1b75bb",
            "include_zero": False,
        },
        {
            "title": "Completed episode return",
            "x": [row["end_step"] for row in episodes],
            "y": [row["return"] for row in episodes],
            "outcomes": [row["outcome"] for row in episodes],
            "color": "#2ca02c",
            "include_zero": False,
        },
        {
            "title": "Completed episode length [steps]",
            "x": [row["end_step"] for row in episodes],
            "y": [row["length"] for row in episodes],
            "outcomes": [row["outcome"] for row in episodes],
            "color": "#2ca02c",
            "include_zero": True,
        },
    )
    width = 1080
    header_height = 88
    panel_height = 245
    height = header_height + panel_height * len(panels) + 38
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#202124}",
        ".title{font-size:24px;font-weight:650}.panel-title{font-size:15px;font-weight:600}",
        ".axis{stroke:#5f6368;stroke-width:1}.grid{stroke:#dadce0;stroke-width:1}",
        ".tick{font-size:11px;fill:#5f6368}.legend{font-size:12px}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="64" y="36">Training curve</text>',
        (
            f'<text class="tick" x="64" y="60">actual environment steps: '
            f"{total_steps}; records: {len(records)}; completed episodes: "
            f"{len(episodes)}</text>"
        ),
        '<circle cx="706" cy="38" r="5" fill="#2ca02c"/>',
        '<text class="legend" x="718" y="42">truncated (time limit)</text>',
        '<circle cx="884" cy="38" r="5" fill="#d93025"/>',
        '<text class="legend" x="896" y="42">terminated (safety)</text>',
    ]
    for index, panel in enumerate(panels):
        parts.extend(
            _draw_panel(
                panel,
                top=header_height + index * panel_height,
                width=width,
                total_steps=total_steps,
                show_x_label=index == len(panels) - 1,
            )
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _validate_curve(curve: Mapping[str, Any]) -> None:
    if curve.get("schema_version") != TRAINING_CURVE_SCHEMA_VERSION:
        raise ValueError(
            "training curve schema_version must be "
            f"{TRAINING_CURVE_SCHEMA_VERSION!r}"
        )
    record_every = curve.get("record_every")
    initial_steps = curve.get("initial_steps")
    actual_steps = curve.get("actual_steps")
    if isinstance(record_every, bool) or not isinstance(record_every, int):
        raise TypeError("training curve record_every must be a positive integer")
    if record_every <= 0:
        raise ValueError("training curve record_every must be a positive integer")
    if isinstance(initial_steps, bool) or not isinstance(initial_steps, int):
        raise TypeError("training curve initial_steps must be a non-negative integer")
    if initial_steps < 0:
        raise ValueError("training curve initial_steps must be a non-negative integer")
    if isinstance(actual_steps, bool) or not isinstance(actual_steps, int):
        raise TypeError("training curve actual_steps must be a positive integer")
    if actual_steps <= initial_steps:
        raise ValueError("training curve actual_steps must exceed initial_steps")
    records = curve.get("records")
    episodes = curve.get("episodes")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("training curve records must be a sequence")
    if not records:
        raise ValueError("training curve records must not be empty")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise TypeError("training curve episodes must be a sequence")
    previous_end = initial_steps
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("training curve record must be a mapping")
        start = _integer_field(row, "start_step", minimum=0)
        end = _integer_field(row, "end_step", minimum=1)
        transitions = _integer_field(row, "transition_count", minimum=1)
        if start != previous_end or end <= start or transitions != end - start:
            raise ValueError("training curve records must form contiguous step windows")
        for name in (
            "mean_reward",
            "reward_std",
            "minimum_reward",
            "maximum_reward",
        ):
            _finite_field(row, name)
        for name in (
            "completed_episodes",
            "terminated_episodes",
            "truncated_episodes",
        ):
            _integer_field(row, name, minimum=0)
        if (
            int(row["terminated_episodes"]) + int(row["truncated_episodes"])
            != int(row["completed_episodes"])
        ):
            raise ValueError("training curve episode counts are inconsistent")
        previous_end = end
    if previous_end != actual_steps:
        raise ValueError("training curve records must end at actual_steps")
    previous_episode_end = initial_steps
    for row in episodes:
        if not isinstance(row, Mapping):
            raise TypeError("training curve episode must be a mapping")
        end = _integer_field(row, "end_step", minimum=1)
        _integer_field(row, "length", minimum=1)
        _finite_field(row, "return")
        if end < previous_episode_end or end > actual_steps:
            raise ValueError("training curve episode end_step is out of order")
        if row.get("outcome") not in {"terminated", "truncated"}:
            raise ValueError(
                "training curve episode outcome must be terminated or truncated"
            )
        previous_episode_end = end


def _integer_field(row: Mapping[str, Any], name: str, *, minimum: int) -> int:
    if name not in row:
        raise ValueError(f"training curve row is missing {name!r}")
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"training curve {name} must be an integer")
    if value < minimum:
        raise ValueError(f"training curve {name} must be at least {minimum}")
    return value


def _finite_field(row: Mapping[str, Any], name: str) -> float:
    if name not in row:
        raise ValueError(f"training curve row is missing {name!r}")
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"training curve {name} must be finite")
    return value


def _draw_panel(panel, *, top, width, total_steps, show_x_label):
    left = 92.0
    right = float(width - 38)
    chart_top = float(top + 36)
    bottom = float(top + 202)
    x = np.asarray(panel["x"], dtype=float)
    y = np.asarray(panel["y"], dtype=float)
    y_min, y_max = _plot_range(y, include_zero=panel["include_zero"])
    parts = [
        (
            f'<text class="panel-title" x="{left:.0f}" y="{top + 20}">'
            f"{html.escape(str(panel['title']))}</text>"
        )
    ]
    for index in range(5):
        fraction = index / 4.0
        y_value = y_max - fraction * (y_max - y_min)
        pixel = chart_top + fraction * (bottom - chart_top)
        parts.extend(
            [
                f'<line class="grid" x1="{left:.2f}" y1="{pixel:.2f}" x2="{right:.2f}" y2="{pixel:.2f}"/>',
                (
                    f'<text class="tick" text-anchor="end" x="{left - 9:.2f}" '
                    f'y="{pixel + 4:.2f}">{_format_number(y_value)}</text>'
                ),
            ]
        )
    for index in range(5):
        fraction = index / 4.0
        step = total_steps * fraction
        pixel = left + fraction * (right - left)
        parts.extend(
            [
                f'<line class="grid" x1="{pixel:.2f}" y1="{chart_top:.2f}" x2="{pixel:.2f}" y2="{bottom:.2f}"/>',
                (
                    f'<text class="tick" text-anchor="middle" x="{pixel:.2f}" '
                    f'y="{bottom + 18:.2f}">{_format_step(step)}</text>'
                ),
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}"/>',
            f'<line class="axis" x1="{left:.2f}" y1="{chart_top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}"/>',
        ]
    )
    if x.size:
        points = " ".join(
            f"{_map_x(x_value, total_steps, left, right):.2f},"
            f"{_map_y(y_value, y_min, y_max, chart_top, bottom):.2f}"
            for x_value, y_value in zip(x, y)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{panel["color"]}" stroke-width="2.25"/>'
        )
        outcomes = panel["outcomes"]
        for index, (x_value, y_value) in enumerate(zip(x, y)):
            color = panel["color"]
            if outcomes is not None:
                color = "#d93025" if outcomes[index] == "terminated" else "#2ca02c"
            parts.append(
                f'<circle cx="{_map_x(x_value, total_steps, left, right):.2f}" '
                f'cy="{_map_y(y_value, y_min, y_max, chart_top, bottom):.2f}" '
                f'r="3.6" fill="{color}"/>'
            )
    else:
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{(left + right) / 2:.2f}" '
            f'y="{(chart_top + bottom) / 2:.2f}">no completed episodes</text>'
        )
    if show_x_label:
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{(left + right) / 2:.2f}" '
            f'y="{bottom + 36:.2f}">environment steps</text>'
        )
    return parts


def _plot_range(values: np.ndarray, *, include_zero: bool) -> tuple[float, float]:
    return plot_range(
        values,
        include_zero=include_zero,
        padding_fraction=0.08,
        floor_zero=True,
    )


def _map_x(value, maximum, left, right):
    return map_value(value, 0.0, maximum, left, right)


def _map_y(value, minimum, maximum, top, bottom):
    return map_value(value, minimum, maximum, bottom, top)


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 10000 or (0 < absolute < 0.001):
        return f"{value:.2e}"
    if absolute >= 100:
        return f"{value:.1f}"
    if absolute >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _format_step(value: float) -> str:
    rounded = int(round(value))
    return f"{rounded:,}"


__all__ = [
    "plot_training_curve",
    "TRAINING_CURVE_SCHEMA_VERSION",
]

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
from .evaluate import DEFAULT_VALIDATION_SETTLING_FRACTION, _validation_summary


TRAINING_CURVE_SCHEMA_VERSION = "aiogym.training_curve.v2"


def plot_training_curve(
    curve: Mapping[str, Any] | str | Path,
    *,
    output: str | Path,
    evaluation_history: Mapping[str, Any] | str | Path | None = None,
    control_dt: float | None = None,
    settling_fraction: float = DEFAULT_VALIDATION_SETTLING_FRACTION,
) -> Path:
    """Plot fixed-case validation; retain training samples in the input artifact.

    ``control_dt`` is required with validation history. Control success means
    safe full-horizon completion and remaining within the scenario's existing
    settling tolerance throughout the final ``settling_fraction`` of the planned control steps (rounded up).
    """

    payload = _load_curve(curve)
    _validate_curve(payload)
    window = _positive_number("settling_fraction", settling_fraction)
    if window > 1:
        raise ValueError("settling_fraction must be at most 1")
    history = None if evaluation_history is None else _load_curve(evaluation_history)
    rows = [] if history is None else _validation_rows(
        history, payload, control_dt=control_dt, settling_fraction=window,
    )
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _training_curve_svg(payload, history, rows, window), encoding="utf-8",
    )
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


def _positive_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _validation_rows(history, curve, *, control_dt, settling_fraction):
    dt = _positive_number("control_dt", control_dt)
    seeds = history.get("seeds")
    records = history.get("records")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError("validation seeds must be distinct non-negative integers")
    if not isinstance(records, list) or not records:
        raise ValueError("validation records must be a non-empty list")
    previous_step = int(curve["initial_steps"]) - 1
    cases = None
    rows = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("validation record must be a mapping")
        step = _integer_field(record, "step", minimum=0)
        if not previous_step < step <= curve["actual_steps"]:
            raise ValueError("validation steps must increase within the training interval")
        previous_step = step
        episodes = record.get("episodes")
        if not isinstance(episodes, list) or not all(
            isinstance(episode, Mapping) for episode in episodes
        ) or [episode.get("seed") for episode in episodes] != seeds:
            raise ValueError("each validation record must contain the same ordered seeds")
        current_cases = []
        for episode in episodes:
            spec = episode.get("episode_spec")
            metrics = episode.get("metrics")
            if not isinstance(spec, Mapping) or not isinstance(metrics, Mapping):
                raise ValueError("validation episodes require episode_spec and metrics")
            horizon = _integer_field(spec, "horizon", minimum=1)
            length = _integer_field(episode, "length", minimum=1)
            if length > horizon:
                raise ValueError("validation episode length exceeds its horizon")
            for flag in ("terminated", "truncated"):
                if not isinstance(episode.get(flag), bool):
                    raise TypeError(f"validation {flag} must be bool")
            current_cases.append((spec, episode.get("runtime_variation", {})))
            _finite_field(episode, "return")
        if cases is None:
            cases = current_cases
        elif cases != current_cases:
            raise ValueError("validation cases must remain fixed across records")
        summary = _validation_summary(
            episodes, control_dt=dt, settling_fraction=settling_fraction,
        )
        rows.append({
            "step": step,
            **summary,
            "safe_completion": summary["safe_completion"] * 100,
            "control_success": summary["control_success"] * 100,
        })
    if rows[0]["step"] != curve["initial_steps"] or rows[-1]["step"] != curve["actual_steps"]:
        raise ValueError("validation must cover the initial and final training steps")
    best_step = _integer_field(history, "best_step", minimum=0)
    if best_step not in [row["step"] for row in rows]:
        raise ValueError("best_step must identify a validation record")
    return rows


def _training_curve_svg(curve, history, rows, settling_fraction) -> str:
    total_steps = int(curve["actual_steps"])
    original_selection = history is not None and history.get("settling_fraction") != settling_fraction
    panels = (
        {
            "title": "Fixed-validation episode return (symlog axis; higher is better)",
            "series": (
                ("mean_return", "Mean return", "#1b75bb", ""),
                ("p10_return", "P10 return", "#b36716", "6 4"),
            ),
            "percent": False,
        },
        {
            "title": "Fixed-validation safety and control success",
            "series": (
                ("safe_completion", "Safe completion", "#64748b", "6 4"),
                ("control_success", "Control success", "#16835b", ""),
            ),
            "percent": True,
        },
    )
    width = 1080
    header_height = 110
    panel_height = 270
    height = header_height + panel_height * len(panels) + 84
    description = (
        "No validation data. Enable evaluation_env and evaluate_every during training."
        if history is None else
        f"{len(history['seeds'])} fixed validation cases | {len(rows)} evaluations | "
        f"one training run | selected best: {history['best_step']:,} steps"
        + (" (original selection; not reselected)" if original_selection else "")
    )
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
        '<text class="title" x="64" y="36">Validation learning curve</text>',
        (
            f'<text class="legend" x="64" y="62">{html.escape(description)}</text>'
        ),
        (
            f'<text class="tick" x="64" y="84">Control success: safe full episode + '
            f'last {settling_fraction:.0%} of planned control steps continuously within scenario settling tolerances.</text>'
        ),
    ]
    for index, panel in enumerate(panels):
        parts.extend(
            _draw_panel(
                panel,
                top=header_height + index * panel_height,
                width=width,
                total_steps=total_steps,
                initial_steps=curve["initial_steps"],
                rows=rows,
                best_step=None if history is None else history["best_step"],
                show_x_label=index == len(panels) - 1,
            )
        )
    parts.extend([
        f'<text class="tick" x="64" y="{height - 52}">Return axis: symmetric log, linear within [-1, 1]; '
        'tick labels and reported values are raw episode returns.</text>',
        f'<text class="tick" x="64" y="{height - 34}">P10 is the lower case-return percentile, not a confidence interval. '
        'Rates use all fixed cases, including failures.</text>',
        f'<text class="tick" x="64" y="{height - 16}">Training interaction rewards remain in training_curve.json; '
        'validation episode details remain in evaluation_history.json.</text>',
        "</svg>",
    ])
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


def _draw_panel(
    panel, *, top, width, initial_steps, total_steps, rows, best_step, show_x_label,
):
    left = 92.0
    right = float(width - 38)
    chart_top = float(top + 62)
    bottom = float(top + 224)
    values = [row[key] for key, *_ in panel["series"] for row in rows]
    if panel["percent"]:
        y_min, y_max = 0.0, 100.0
        ticks = list(np.linspace(0, 100, 5))
    else:
        y_min, y_max = plot_range(
            [_return_coordinate(value) for value in values],
            include_zero=True, padding_fraction=0.08,
        )
        if values and max(abs(value) for value in values) > 1:
            powers = range(math.ceil(math.log10(max(abs(value) for value in values))) + 1)
            ticks = sorted([0.0] + [sign * 10.0 ** power for power in powers for sign in (-1, 1)])
            ticks = [value for value in ticks if y_min <= _return_coordinate(value) <= y_max]
        else:
            ticks = list(np.linspace(y_min, y_max, 5))

    def map_x(value):
        return map_value(value, initial_steps, total_steps, left, right)

    def map_y(value):
        coordinate = value if panel["percent"] else _return_coordinate(value)
        return _map_y(coordinate, y_min, y_max, chart_top, bottom)

    parts = [
        (
            f'<text class="panel-title" x="{left:.0f}" y="{top + 20}">'
            f"{html.escape(str(panel['title']))}</text>"
        )
    ]
    for index, (key, label, color, dash) in enumerate(panel["series"]):
        legend_x = left + index * 300
        suffix = "" if not rows else (
            f"  (final: {rows[-1][key]:.0f}%)" if panel["percent"]
            else f"  (final: {_format_number(rows[-1][key])})"
        )
        parts.extend([
            f'<line x1="{legend_x}" x2="{legend_x + 24}" y1="{top + 40}" y2="{top + 40}" '
            f'stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>',
            f'<text class="legend" x="{legend_x + 32}" y="{top + 44}">{label}{suffix}</text>',
        ])
    for y_value in ticks:
        pixel = map_y(y_value)
        parts.extend(
            [
                f'<line class="grid" x1="{left:.2f}" y1="{pixel:.2f}" x2="{right:.2f}" y2="{pixel:.2f}"/>',
                (
                    f'<text class="tick" text-anchor="end" x="{left - 9:.2f}" '
                    f'y="{pixel + 4:.2f}">{f"{y_value:.0f}%" if panel["percent"] else _format_number(y_value)}</text>'
                ),
            ]
        )
    for index in range(5):
        fraction = index / 4.0
        step = initial_steps + (total_steps - initial_steps) * fraction
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
    if best_step is not None:
        pixel = map_x(best_step)
        parts.extend([
            f'<line x1="{pixel:.2f}" x2="{pixel:.2f}" y1="{chart_top}" y2="{bottom}" '
            'stroke="#9b7bb5" stroke-dasharray="3 4"/>',
            f'<text class="tick" text-anchor="end" x="{right}" y="{top + 20}">'
            f'Selected best: {_format_step(best_step)}</text>',
        ])
    for key, label, color, dash in panel["series"]:
        points = " ".join(
            f"{map_x(row['step']):.2f},"
            f"{map_y(row[key]):.2f}"
            for row in rows
        )
        if rows:
            parts.append(
                f'<polyline data-series="{key}" points="{points}" fill="none" '
                f'stroke="{color}" stroke-width="2.25" stroke-dasharray="{dash}"/>'
            )
        for row in rows:
            parts.append(
                f'<circle data-series="{key}" data-step="{row["step"]}" data-value="{row[key]}" '
                f'cx="{map_x(row["step"]):.2f}" '
                f'cy="{map_y(row[key]):.2f}" '
                f'r="3" fill="{color}"><title>{label}: {row[key]:.6g}; '
                f'step {row["step"]}</title></circle>'
            )
    if not rows:
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{(left + right) / 2:.2f}" '
            f'y="{(chart_top + bottom) / 2:.2f}">No fixed-validation measurements</text>'
        )
    if show_x_label:
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{(left + right) / 2:.2f}" '
            f'y="{bottom + 36:.2f}">training environment steps</text>'
        )
    return parts


def _return_coordinate(value):
    return value if abs(value) <= 1 else math.copysign(1 + math.log10(abs(value)), value)


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

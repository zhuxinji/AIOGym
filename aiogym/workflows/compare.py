"""Descriptively compare policies on the same ordered seeds."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiogym.core.io import write_json

from .evaluate import evaluate


COMPARISON_SCHEMA_VERSION = "aiogym.comparison.v3"
_COLORS = (
    "#d95f02",
    "#1b75bb",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e7298a",
    "#17becf",
    "#7f7f7f",
)


def compare_policies(
    *,
    env,
    policies: Mapping[str, Any],
    seeds: Sequence[int],
    max_steps: int | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Compare policies and persist one JSON/SVG benchmark report.

    Benchmark environments default to ``runs/<scenario>/<benchmark>`` and
    replace only the two managed comparison artifacts. Comparisons outside a
    Benchmark require an explicit empty output directory.
    """
    if not isinstance(policies, Mapping) or len(policies) < 2:
        raise ValueError("policies must be a mapping with at least two entries")
    labels = tuple(policies)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("policy labels must be non-empty strings")
    output_directory, overwrite = _prepare_output_directory(env, output)

    evaluations = {
        label: evaluate(
            env=env,
            policy=policies[label],
            seeds=seeds,
            max_steps=max_steps,
        )
        for label in labels
    }
    first = evaluations[labels[0]]
    ordered_seeds = first["seeds"]
    if any(result["seeds"] != ordered_seeds for result in evaluations.values()):
        raise ValueError("all policy evaluations must use identical ordered seeds")
    ranking_metrics = first["ranking_metrics"]
    if any(
        result["ranking_metrics"] != ranking_metrics for result in evaluations.values()
    ):
        raise ValueError("all policy evaluations must use identical ranking metrics")
    trajectory_schema = first["trajectory_schema"]
    if any(
        result["trajectory_schema"] != trajectory_schema
        for result in evaluations.values()
    ):
        raise ValueError("all policy evaluations must use identical trajectory schemas")

    def ranking_key(label):
        result = evaluations[label]
        values = []
        for metric in ranking_metrics:
            median = result["aggregate"][metric["name"]]["median"]
            values.append(median if metric["direction"] == "minimize" else -median)
        return (*values, label)

    ordering = sorted(labels, key=ranking_key)
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "environment": dict(first["environment"]),
        "trajectory_schema": trajectory_schema,
        "seeds": list(ordered_seeds),
        "max_steps": first["max_steps"],
        "ranking_metrics": ranking_metrics,
        "ordering": ordering,
        "evaluations": evaluations,
    }
    svg_path = output_directory / "comparison.svg"
    svg_path.write_text(_comparison_svg(result), encoding="utf-8")
    write_json(
        output_directory / "comparison.json",
        result,
        overwrite=overwrite,
    )
    return result


def _prepare_output_directory(env, output) -> tuple[Path, bool]:
    overwrite = output is None
    directory = _default_output_directory(env) if overwrite else Path(output)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"comparison output is not a directory: {directory}")
    entries = tuple(directory.iterdir()) if directory.exists() else ()
    if not overwrite and entries:
        raise FileExistsError(
            f"refusing to create comparison in non-empty directory: {directory}"
        )
    managed_names = {"comparison.json", "comparison.svg"}
    unexpected = sorted(
        path.name for path in entries if path.name not in managed_names
    )
    if unexpected:
        raise FileExistsError(
            f"default comparison directory contains unmanaged entries: {unexpected}"
        )
    invalid = sorted(
        path.name for path in entries if path.is_symlink() or not path.is_file()
    )
    if invalid:
        raise FileExistsError(
            f"default comparison artifacts must be files: {invalid}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory, overwrite


def _default_output_directory(env) -> Path:
    base_env = env.unwrapped
    if base_env.benchmark is None:
        raise ValueError(
            "output is required when comparing policies outside a benchmark"
        )
    scenario_id = base_env.scenario.id.replace("_", "-")
    benchmark_id = base_env.benchmark.id
    return Path("runs") / scenario_id / benchmark_id


def _comparison_svg(result) -> str:
    labels = tuple(result["evaluations"])
    schema = result["trajectory_schema"]
    panels = []
    for index, row in enumerate(schema["output"]):
        y_limits = (
            _schema_bounds(row)
            if "level" in str(row["name"]).lower()
            else None
        )
        panels.append(
            _series_panel(
                result,
                labels,
                title=f"Output: {_schema_label(row)}",
                field="output",
                index=index,
                reference_field="reference",
                y_limits=y_limits,
            )
        )
    for index, row in enumerate(schema["action"]):
        panels.append(
            _series_panel(
                result,
                labels,
                title=f"Applied action: {_schema_label(row)}",
                field="applied_action",
                index=index,
                y_limits=_schema_bounds(row),
            )
        )
    first_summary = result["evaluations"][labels[0]]["trajectory_summary"]
    for name, band in first_summary["disturbance"].items():
        values = np.asarray(band["median"], dtype=float)
        if values.size and float(np.max(values) - np.min(values)) > 1e-12:
            panels.append(
                _single_series_panel(
                    result,
                    labels[0],
                    title=f"Disturbance: {name}",
                    field="disturbance",
                    mapping_name=name,
                )
            )
    panels.append(
        _series_panel(
            result,
            labels,
            title="Minimum safety margin",
            field="minimum_safety_margin",
            horizontal=0.0,
        )
    )

    width = 1080
    header_height = 100
    panel_height = 225
    return_panel_height = max(285, 100 + 34 * len(labels))
    height = header_height + panel_height * len(panels) + return_panel_height
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
        (
            f'<text class="title" x="64" y="38">'
            f"{html.escape(str(result['environment']['scenario']))} policy comparison"
            "</text>"
        ),
        (
            f'<text class="tick" x="64" y="62">seeds: '
            f"{html.escape(', '.join(str(seed) for seed in result['seeds']))}"
            "</text>"
        ),
    ]
    legend_x = 430
    for index, label in enumerate(labels):
        x = legend_x + (index % 4) * 145
        y = 34 + (index // 4) * 22
        color = _COLORS[index % len(_COLORS)]
        parts.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<text class="legend" x="{x + 30}" y="{y + 4}">{html.escape(label)}</text>',
            ]
        )
    reference_index = len(labels)
    reference_x = legend_x + (reference_index % 4) * 145
    reference_y = 34 + (reference_index // 4) * 22
    parts.extend(
        [
            f'<line x1="{reference_x}" y1="{reference_y}" x2="{reference_x + 24}" y2="{reference_y}" stroke="#111111" stroke-width="2" stroke-dasharray="7 5"/>',
            f'<text class="legend" x="{reference_x + 30}" y="{reference_y + 4}">reference</text>',
        ]
    )
    top = header_height
    for index, panel in enumerate(panels):
        parts.extend(_draw_series_panel(panel, top, index))
        top += panel_height
    parts.extend(_draw_return_distribution(result, labels, top))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _schema_label(row) -> str:
    name = str(row["name"])
    return name if "unit" not in row else f"{name} [{row['unit']}]"


def _schema_bounds(row):
    if "low" not in row or "high" not in row:
        return None
    low = float(row["low"])
    high = float(row["high"])
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("finite plot schema bounds must satisfy low < high")
    return low, high


def _series_panel(
    result,
    labels,
    *,
    title,
    field,
    index=None,
    reference_field=None,
    horizontal=None,
    y_limits=None,
):
    series = []
    for label_index, label in enumerate(labels):
        evaluation = result["evaluations"][label]
        series.append(
            {
                "label": label,
                "color": _COLORS[label_index % len(_COLORS)],
                "time": evaluation["trajectory_summary"]["physical_time"]["median"],
                "band": evaluation["trajectory_summary"][field],
                "index": index,
            }
        )
    reference = None
    if reference_field is not None:
        evaluation = result["evaluations"][labels[0]]
        reference = {
            "time": evaluation["trajectory_summary"]["physical_time"]["median"],
            "band": evaluation["trajectory_summary"][reference_field],
            "index": index,
        }
    return {
        "title": title,
        "series": series,
        "reference": reference,
        "horizontal": horizontal,
        "y_limits": y_limits,
    }


def _single_series_panel(
    result,
    label,
    *,
    title,
    field,
    mapping_name,
):
    evaluation = result["evaluations"][label]
    return {
        "title": title,
        "series": [
            {
                "label": mapping_name,
                "color": "#5f6368",
                "time": evaluation["trajectory_summary"]["physical_time"]["median"],
                "band": evaluation["trajectory_summary"][field][mapping_name],
                "index": None,
            }
        ],
        "reference": None,
        "horizontal": None,
        "y_limits": None,
    }


def _draw_series_panel(panel, top, panel_index):
    left = 76.0
    right = 1040.0
    chart_top = float(top + 36)
    bottom = float(top + 180)
    all_x = []
    all_y = []
    resolved_series = []
    for series in panel["series"]:
        time = _finite_array("trajectory time", series["time"])
        median = _band_component(series["band"], "median", series["index"])
        minimum = _band_component(series["band"], "min", series["index"])
        maximum = _band_component(series["band"], "max", series["index"])
        _matching_lengths(time, median, minimum, maximum)
        all_x.extend(time.tolist())
        all_y.extend(minimum.tolist())
        all_y.extend(maximum.tolist())
        resolved_series.append((series, time, median, minimum, maximum))
    reference = panel["reference"]
    resolved_reference = None
    if reference is not None:
        time = _finite_array("reference time", reference["time"])
        values = _band_component(reference["band"], "median", reference["index"])
        _matching_lengths(time, values)
        all_x.extend(time.tolist())
        all_y.extend(values.tolist())
        resolved_reference = (time, values)
    horizontal = panel["horizontal"]
    if horizontal is not None:
        all_y.append(float(horizontal))
    time_values = _finite_array("trajectory time", all_x).reshape(-1)
    if np.any(time_values < 0.0):
        raise ValueError("trajectory time must be non-negative")
    x_min = 0.0
    x_max = float(np.max(time_values))
    if x_max <= x_min:
        raise ValueError("trajectory time must contain a positive value")
    if panel["y_limits"] is None:
        y_min, y_max = _plot_range(all_y, include_zero=False)
    else:
        y_min, y_max = panel["y_limits"]
    clip_id = f"panel-clip-{panel_index}"

    parts = [
        f'<text class="panel-title" x="{left}" y="{top + 20}">{html.escape(panel["title"])}</text>',
        f'<defs><clipPath id="{clip_id}"><rect x="{left}" y="{chart_top}" width="{right - left}" height="{bottom - chart_top}"/></clipPath></defs>',
    ]
    for tick in range(5):
        fraction = tick / 4
        x = left + fraction * (right - left)
        value = x_min + fraction * (x_max - x_min)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{chart_top:.2f}" x2="{x:.2f}" y2="{bottom:.2f}"/>',
                f'<text class="tick" text-anchor="middle" x="{x:.2f}" y="{bottom + 18:.2f}">{_number(value)}</text>',
            ]
        )
    for tick in range(4):
        fraction = tick / 3
        y = bottom - fraction * (bottom - chart_top)
        value = y_min + fraction * (y_max - y_min)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>',
                f'<text class="tick" text-anchor="end" x="{left - 8}" y="{y + 4:.2f}">{_number(value)}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{chart_top}" x2="{left}" y2="{bottom}"/>',
            f'<text class="tick" text-anchor="middle" x="{(left + right) / 2:.2f}" y="{bottom + 34}">time [s]</text>',
            f'<g clip-path="url(#{clip_id})">',
        ]
    )
    if horizontal is not None:
        y = _map_y(float(horizontal), y_min, y_max, chart_top, bottom)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#c62828" stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
    if resolved_reference is not None:
        time, values = resolved_reference
        points = _polyline_points(
            time,
            values,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            right,
            chart_top,
            bottom,
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#111111" stroke-width="1.8" stroke-dasharray="7 5"/>'
        )
    for series, time, median, minimum, maximum in resolved_series:
        color = series["color"]
        polygon = _band_polygon(
            time,
            minimum,
            maximum,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            right,
            chart_top,
            bottom,
        )
        points = _polyline_points(
            time,
            median,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            right,
            chart_top,
            bottom,
        )
        parts.extend(
            [
                f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.16" stroke="none"/>',
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>',
            ]
        )
    parts.append("</g>")
    return parts


def _draw_return_distribution(result, labels, top):
    left = 160.0
    right = 1040.0
    chart_top = float(top + 42)
    row_height = 34.0
    bottom = chart_top + row_height * len(labels)
    distributions = {
        label: _finite_array(
            "return distribution",
            result["evaluations"][label]["return_distribution"],
        )
        for label in labels
    }
    all_values = np.concatenate(tuple(distributions.values()))
    value_min, value_max = _plot_range(all_values.tolist(), include_zero=False)
    parts = [
        f'<text class="panel-title" x="76" y="{top + 22}">Cumulative return by policy</text>',
        f'<text class="tick" x="430" y="{top + 22}">circles: evaluation seeds; diamond: median</text>',
    ]
    for tick in range(5):
        fraction = tick / 4
        x = left + fraction * (right - left)
        value = value_min + fraction * (value_max - value_min)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{chart_top:.2f}" x2="{x:.2f}" y2="{bottom:.2f}"/>',
                f'<text class="tick" text-anchor="middle" x="{x:.2f}" y="{bottom + 18:.2f}">{_number(value)}</text>',
            ]
        )
    for label_index, label in enumerate(labels):
        values = distributions[label]
        y = chart_top + (label_index + 0.5) * row_height
        color = _COLORS[label_index % len(_COLORS)]
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>',
                f'<text class="legend" text-anchor="end" x="{left - 12}" y="{y + 4:.2f}">{html.escape(label)}</text>',
            ]
        )
        offsets = (
            np.linspace(-6.0, 6.0, len(values))
            if len(values) > 1
            else np.zeros(1)
        )
        for value, offset in zip(values, offsets):
            x = _map_x(value, value_min, value_max, left, right)
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y + offset:.2f}" r="4" fill="{color}" fill-opacity="0.48"/>'
            )
        median_x = _map_x(
            float(np.median(values)), value_min, value_max, left, right
        )
        diamond = " ".join(
            (
                f"{median_x:.2f},{y - 7:.2f}",
                f"{median_x + 7:.2f},{y:.2f}",
                f"{median_x:.2f},{y + 7:.2f}",
                f"{median_x - 7:.2f},{y:.2f}",
            )
        )
        parts.append(
            f'<polygon points="{diamond}" fill="{color}" stroke="#202124" stroke-width="1"/>'
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<text class="tick" text-anchor="middle" x="{(left + right) / 2:.2f}" y="{bottom + 36}">episode return (higher is better)</text>',
        ]
    )
    return parts


def _band_component(band, name, index):
    values = _finite_array(f"trajectory band {name}", band[name])
    if index is None:
        if values.ndim != 1:
            raise ValueError("scalar trajectory bands must be one-dimensional")
        return values
    if values.ndim != 2 or index >= values.shape[1]:
        raise ValueError("vector trajectory band does not match its schema")
    return values[:, index]


def _finite_array(name, values):
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _matching_lengths(*arrays):
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("trajectory plot series must have matching lengths")


def _plot_range(values, *, include_zero):
    array = _finite_array("plot values", values).reshape(-1)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if include_zero:
        minimum = min(0.0, minimum)
        maximum = max(0.0, maximum)
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        padding = max(1.0, abs(minimum) * 0.05)
    else:
        padding = 0.05 * (maximum - minimum)
    return minimum - padding, maximum + padding


def _map_x(value, minimum, maximum, left, right):
    return left + (float(value) - minimum) / (maximum - minimum) * (right - left)


def _map_y(value, minimum, maximum, top, bottom):
    return bottom - (float(value) - minimum) / (maximum - minimum) * (bottom - top)


def _polyline_points(
    x,
    y,
    x_min,
    x_max,
    y_min,
    y_max,
    left,
    right,
    top,
    bottom,
):
    return " ".join(
        f"{_map_x(x_value, x_min, x_max, left, right):.2f},"
        f"{_map_y(y_value, y_min, y_max, top, bottom):.2f}"
        for x_value, y_value in zip(x, y)
    )


def _band_polygon(
    x,
    minimum,
    maximum,
    x_min,
    x_max,
    y_min,
    y_max,
    left,
    right,
    top,
    bottom,
):
    upper = _polyline_points(
        x,
        maximum,
        x_min,
        x_max,
        y_min,
        y_max,
        left,
        right,
        top,
        bottom,
    )
    lower = _polyline_points(
        x[::-1],
        minimum[::-1],
        x_min,
        x_max,
        y_min,
        y_max,
        left,
        right,
        top,
        bottom,
    )
    return f"{upper} {lower}"


def _number(value):
    number = float(value)
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude >= 1e4 or magnitude < 1e-3):
        return f"{number:.2e}"
    return f"{number:.3g}"


__all__ = ["COMPARISON_SCHEMA_VERSION", "compare_policies"]

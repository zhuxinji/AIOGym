"""Render policy-comparison trajectories as a dependency-free SVG."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping

import numpy as np

from ._svg import map_value, plot_range
from .evaluate import DEFAULT_VALIDATION_SETTLING_FRACTION, _validation_summary


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


def render_trajectory_svg(result, *, title: str | None = None) -> str:
    labels = tuple(result["evaluations"])
    settling_fraction = result.get("settling_fraction", DEFAULT_VALIDATION_SETTLING_FRACTION)
    schema = result["trajectory_schema"]
    time_unit = str(schema["time_unit"])
    resolved_title = title or (
        f"{result['environment']['scenario']} / "
        f"{result['environment']['benchmark'] or result['environment']['reward']} comparison"
    )
    baseline = next(
        (label for label in labels if result["evaluations"][label]["policy"].get("id") == "pid"),
        labels[0],
    )
    paired_labels = tuple(label for label in labels if label != baseline)
    episodes = {
        label: {episode["seed"]: episode for episode in result["evaluations"][label]["episodes"]}
        for label in labels
    }
    case_summaries = {
        label: {
            seed: _validation_summary(
                [episode], control_dt=result["environment"]["control_dt"],
                settling_fraction=settling_fraction,
            ) if {
                "constraint_violations", "settling_time"
            } <= episode["metrics"].keys() else None
            for seed, episode in cases.items()
        }
        for label, cases in episodes.items()
    }
    temperature_indices = [
        index for index, row in enumerate(schema["output"]) if row.get("unit") == "degC"
    ]
    temperature_limits = None
    if temperature_indices:
        values = []
        for label in labels:
            trajectory = _plotted_trajectory(result, label)
            values.extend(np.asarray(trajectory["output"])[:, temperature_indices].ravel())
        values.extend(
            np.asarray(_plotted_trajectory(result, labels[0])["reference"])[:, temperature_indices].ravel()
        )
        temperature_limits = _plot_range(values, include_zero=False)
    output_panels = [
        _series_panel(
            result, labels, title=f"Output: {_schema_label(row)}",
            field="output", index=index, reference_field="reference",
            y_limits=_schema_bounds(row) if "level" in row["name"].lower()
            else temperature_limits if index in temperature_indices else None,
        )
        for index, row in enumerate(schema["output"])
    ]
    action_panels = [
        _series_panel(
            result, labels, title=f"Applied action: {_schema_label(row)}",
            field="applied_action", index=index, y_limits=_schema_bounds(row),
        )
        for index, row in enumerate(schema["action"])
    ]
    disturbance_panels = []
    first_trajectory = _plotted_trajectory(result, labels[0])
    for name in schema["disturbance_names"]:
        values = _mapping_values(first_trajectory["disturbance"], name)
        if values.size and float(np.ptp(values)) > 1e-12:
            disturbance_panels.append(_single_series_panel(
                result, labels[0], title=f"Disturbance: {name}",
                field="disturbance", mapping_name=name,
                y_limits=_disturbance_y_limits(values),
            ))
    # Preserve physical-role grouping for the multi-tank interfaces.
    groups = [disturbance_panels[i:i + 3] for i in range(0, len(disturbance_panels), 3)]
    if result["environment"]["scenario"] == "cascade":
        groups.extend((output_panels[:3], output_panels[3:], action_panels[:4], action_panels[4:]))
    elif len(output_panels) == 3 and len(action_panels) == 4:
        groups.extend((output_panels, action_panels))
    elif len(output_panels) == 1:
        groups.extend((output_panels, action_panels))
    elif disturbance_panels:
        for panels in (output_panels, action_panels):
            groups.extend(panels[i:i + 3] for i in range(0, len(panels), 3))
    else:
        groups.extend([
            output_panels[index:index + 1] + action_panels[index:index + 1]
            for index in range(max(len(output_panels), len(action_panels)))
        ])

    seeds = result["seeds"]
    # Keep plot geometry intact inside the inset section bodies.
    width = max(1440, 360 + 150 * len(labels))
    canvas_width = width + 88
    header_height = max(128, 76 + 26 * math.ceil((len(labels) + 1) / 4))
    summary_top = header_height + 225 * len(groups)
    paired_top = summary_top + 220
    height = paired_top + (312 + 48 * len(paired_labels) if paired_labels else 120)
    sections = [
        (header_height, "Shared-case tracking" if paired_labels else "Case tracking",
         f"Seed {result['trajectory_seed']} · outputs, references and applied actions"),
        (summary_top + 78, "Absolute performance", f"All {len(seeds)} evaluation cases"),
    ]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{height}" viewBox="0 0 {canvas_width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#202124}",
        ".title{font-size:34px;font-weight:700}.panel-title{font-size:15px;font-weight:600}",
        ".axis{stroke:#5f6368;stroke-width:1}.grid{stroke:#dadce0;stroke-width:1}",
        ".tick{font-size:11px;fill:#5f6368}.legend{font-size:12px}",
        ".header-legend{font-size:14px;font-weight:550}",
        ".section{font-size:24px;font-weight:700;fill:#263649}.body{font-size:14px}",
        ".section-number{font-size:18px;font-weight:650;fill:#6d7f92}",
        ".section-divider{stroke:#8a99aa;stroke-width:2}",
        ".metric{font-size:16px;font-variant-numeric:tabular-nums}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="64" y="60">{html.escape(resolved_title.upper())}</text>',
        f'<text class="tick" x="64" y="96">trajectory seed: {result["trajectory_seed"]}; evaluation cases: {len(seeds)}; '
        f'baseline: {html.escape(baseline) if paired_labels else "single policy"}</text>',
    ]
    for index, label in enumerate((*labels, "reference")):
        x = width - 664 + index % 4 * 165
        y = 76 + index // 4 * 26
        color = _COLORS[index % len(_COLORS)] if index < len(labels) else "#111111"
        dash = ' stroke-dasharray="7 5"' if index == len(labels) else ""
        parts.extend((
            f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3.5"{dash}/>',
            f'<text class="header-legend" x="{x + 36}" y="{y + 5}">{html.escape(label)}</text>',
        ))
    parts.append('<g data-section="tracking" transform="translate(56 48)">')
    panel_index = 0
    for row, panels in enumerate(groups):
        gap = 24 if len(panels) == 4 else 48
        panel_width = (width - 116 - gap * (len(panels) - 1)) / len(panels)
        for column, panel in enumerate(panels):
            panel["time_unit"] = time_unit
            left = 76.0 + column * (panel_width + gap)
            parts.extend(_draw_series_panel(
                panel, header_height + row * 225, panel_index,
                left=left, right=left + panel_width,
            ))
            panel_index += 1

    parts.extend((
        '</g>',
        '<g data-section="summary" transform="translate(56 102)">',
        f'<rect x="64" y="{summary_top + 44}" width="{width - 104}" height="145" rx="8" fill="#f5f7fa"/>',
    ))
    metric_labels = (
        ("safe_completion", "Safe full-horizon completion"),
        ("control_success", f"Control success (last {settling_fraction:.0%} of steps)"),
        ("mean_return", "Mean return (higher is better)"),
    )
    for row, (_key, label) in enumerate(metric_labels):
        parts.append(f'<text class="body" x="80" y="{summary_top + 102 + row * 35}">{label}</text>')
    column_width = (width - 370) / len(labels)
    for index, label in enumerate(labels):
        x = 330 + (index + 0.5) * column_width
        color = _COLORS[index % len(_COLORS)]
        parts.append(f'<text class="body" x="{x}" y="{summary_top + 68}" text-anchor="middle" '
                     f'style="fill:{color};font-weight:650">{html.escape(label)}</text>')
        for row, (key, _caption) in enumerate(metric_labels):
            if key == "mean_return":
                value = result["evaluations"][label]["aggregate"]["episode_return"]["mean"]
                display = _number(value)
            elif all(case is not None for case in case_summaries[label].values()):
                count = sum(case[key] for case in case_summaries[label].values())
                value = count / len(seeds)
                display = f"{value:.0%} ({int(count)}/{len(seeds)})"
            else:
                value = None
                display = "N/A"
            parts.append(
                f'<text class="metric" data-policy="{html.escape(label, quote=True)}" data-metric="{key}" '
                f'data-value="{value}" x="{x}" y="{summary_top + 102 + row * 35}" text-anchor="middle">{display}</text>'
            )
    parts.append(f'<text class="tick" x="76" y="{summary_top + 210}">'
                 f'Safe = full horizon without violations; control success also holds every scenario settling tolerance for the last {settling_fraction:.0%} of planned control steps (rounded up). '
                 'N/A = required metrics unavailable.</text></g>')

    if paired_labels:
        pairs = {}
        excluded = {}
        for label in paired_labels:
            pairs[label], excluded[label] = [], []
            for seed in seeds:
                episode, reference = episodes[label][seed], episodes[baseline][seed]
                stats, ref_stats = case_summaries[label][seed], case_summaries[baseline][seed]
                if stats is None or ref_stats is None:
                    excluded[label].append(f"seed {seed}: required control metrics unavailable")
                elif not stats["safe_completion"] or not ref_stats["safe_completion"]:
                    excluded[label].append(f"seed {seed}: unsafe or incomplete pair")
                elif "tracking_ise" not in episode["metrics"] or "tracking_ise" not in reference["metrics"]:
                    excluded[label].append(f"seed {seed}: tracking ISE unavailable")
                elif reference["metrics"]["tracking_ise"] == 0:
                    excluded[label].append(f"seed {seed}: ref=0, undefined ratio")
                else:
                    pairs[label].append((
                        seed, episode["metrics"]["tracking_ise"] / reference["metrics"]["tracking_ise"],
                    ))
        all_values = [1.0, *(ratio for rows in pairs.values() for _seed, ratio in rows)]
        logarithmic = min(all_values) > 0
        if logarithmic:
            lower, upper = math.log2(min(all_values)), math.log2(max(all_values))
            span = upper - lower
            if span == 0:
                lower, upper = -1.0, 1.0
                ticks = [0.5, 1.0, 2.0]
            elif span >= 1:
                lower, upper = min(-0.25, math.floor(lower)), max(0.25, math.ceil(upper))
                stride = max(1, math.ceil((upper - lower) / 8))
                ticks = sorted({1.0, *(2.0 ** power for power in range(
                    math.ceil(lower), math.floor(upper) + 1, stride,
                ))})
            else:
                padding = span * 0.12
                lower, upper = lower - padding, upper + padding
                ticks = sorted({1.0, *(2.0 ** value for value in np.linspace(lower, upper, 5))})
                ticks = [tick for tick in ticks if tick == 1 or abs(math.log2(tick)) > (upper - lower) * 0.07]
            axis_label = "Relative ISE · log scale"
        else:
            lower, upper = 0.0, max(all_values) * 1.05
            ticks = sorted({1.0, *np.linspace(lower, upper, 5)})
            # Zero is a valid cost, not a value to discard or replace with epsilon.
            axis_label = "Relative ISE · linear scale (includes zero cost)"
        sections.append((
            paired_top + 128, f"Paired tracking cost vs {'PID' if baseline == 'pid' else baseline}",
            f"{len(seeds)} cases · {axis_label} · largest ratio labeled",
        ))
        left, right = 190.0, float(width - 240)
        precision = max(3, 2 - math.floor(math.log10(upper - lower)))
        chart_top = paired_top + 71
        bottom = chart_top + 48 * len(paired_labels)
        pid_x = _map_x(0.0 if logarithmic else 1.0, lower, upper, left, right)
        parts.extend((
            '<g data-section="paired" transform="translate(56 155)">',
            f'<text class="legend" x="{pid_x - 14}" y="{paired_top + 57}" text-anchor="end" style="fill:#237354">← Better</text>',
            f'<text class="legend" x="{pid_x + 14}" y="{paired_top + 57}" style="fill:#9a5134">Worse →</text>',
        ))
        for tick in ticks:
            x = _map_x(math.log2(tick) if logarithmic else tick, lower, upper, left, right)
            stroke = "#5f6368" if tick == 1 else "#e9edf1"
            dash = ' stroke-dasharray="4 4"' if tick == 1 else ""
            parts.append(f'<line x1="{x}" x2="{x}" y1="{chart_top}" y2="{bottom}" stroke="{stroke}"{dash}/>')
            # Keep nearly identical ratios legible and avoid overlapping zero/one ticks.
            if tick != 1 and abs(x - pid_x) < 45:
                continue
            display = f"{tick:.{precision}g}" if logarithmic and upper - lower < 1 else _number(tick)
            parts.append(f'<text class="legend" x="{x}" y="{bottom + 23}" text-anchor="middle">{display}×</text>')
        for row, label in enumerate(paired_labels):
            center = chart_top + (row + 0.5) * 48
            color = _COLORS[labels.index(label) % len(_COLORS)]
            rows = pairs[label]
            omissions = excluded[label]
            parts.extend((
                f'<g data-policy="{html.escape(label, quote=True)}" data-valid-count="{len(rows)}" data-excluded-count="{len(omissions)}">',
                f'<title>{html.escape("; ".join(omissions)) if omissions else "All cases included"}</title>',
                f'<text class="body" x="76" y="{center + (0 if omissions else 5)}" style="fill:{color};font-weight:650">{html.escape(label)}</text>',
            ))
            if omissions:
                parts.append(f'<text class="tick" x="76" y="{center + 15}">{len(rows)}/{len(seeds)} valid pairs</text>')
            if not rows:
                parts.extend((
                    f'<text class="body" x="{left + 16}" y="{center + 5}" style="fill:#5f6368">N/A — no comparable pairs</text>',
                    "</g>",
                ))
                continue
            values = np.array([ratio for _seed, ratio in rows])
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
            iqr = q3 - q1
            within = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
            low, high = float(within.min()), float(within.max())
            x_low, x_q1, x_median, x_q3, x_high = [
                _map_x(math.log2(value) if logarithmic else value, lower, upper, left, right)
                for value in (low, q1, median, q3, high)
            ]
            parts.extend((
                f'<g data-box-policy="{html.escape(label, quote=True)}" data-q1="{q1:.15g}" data-median="{median:.15g}" data-q3="{q3:.15g}" data-whisker-low="{low:.15g}" data-whisker-high="{high:.15g}">',
                f'<title>{html.escape(label)}: Q1={q1:.6g}×; median={median:.6g}×; Q3={q3:.6g}×; whiskers={low:.6g}–{high:.6g}×; {len(rows)} valid cases</title>',
                f'<line x1="{x_low}" x2="{x_high}" y1="{center}" y2="{center}" stroke="{color}" stroke-width="1.7"/>',
                f'<path d="M {x_low},{center - 7} V {center + 7} M {x_high},{center - 7} V {center + 7}" stroke="{color}" stroke-width="1.7"/>',
                f'<rect x="{x_q1}" y="{center - 11}" width="{x_q3 - x_q1}" height="22" fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="1.7"/>',
                f'<line x1="{x_median}" x2="{x_median}" y1="{center - 11}" y2="{center + 11}" stroke="{color}" stroke-width="3"/>',
                "</g>",
            ))
            worst = int(np.argmax(values))
            for index, (seed, ratio) in enumerate(rows):
                outlier = ratio < low or ratio > high
                if not outlier and index != worst:
                    continue
                x = _map_x(math.log2(ratio) if logarithmic else ratio, lower, upper, left, right)
                episode, reference = episodes[label][seed], episodes[baseline][seed]
                detail = (f"{label}, seed {seed}: {ratio:.8g}× baseline ISE; "
                          f"return={episode['return']:.8g}; baseline return={reference['return']:.8g}")
                parts.append(
                    f'<circle data-policy="{html.escape(label, quote=True)}" data-seed="{seed}" data-ratio="{ratio:.15g}" data-outlier="{str(outlier).lower()}" '
                    f'cx="{x}" cy="{center}" r="3.5" fill="{color}" fill-opacity="0.65"><title>{html.escape(detail)}</title></circle>'
                )
                if index == worst:
                    display = f"{ratio:.{precision}g}" if logarithmic and upper - lower < 1 else _number(ratio)
                    parts.extend((
                        f'<circle cx="{x}" cy="{center}" r="5.5" fill="none" stroke="{color}" stroke-width="1.2" pointer-events="none"/>',
                        f'<text class="legend" x="{x + 14}" y="{center + 4}" style="fill:{color}">seed {seed} · {display}×</text>',
                    ))
            parts.append("</g>")
        parts.append(
            f'<text class="legend" x="76" y="{bottom + 58}">Box: middle 50% · Line: median · Whiskers: within 1.5×IQR · Dots: outliers · Ring: largest ratio · 1× = baseline; lower is better</text></g>'
        )
    for number, (top, heading, detail) in enumerate(sections, start=1):
        parts.extend((
            f'<line class="section-divider" x1="64" x2="{canvas_width - 64}" y1="{top}" y2="{top}"/>',
            f'<text class="section-number" x="64" y="{top + 34}">{number:02d}</text>',
            f'<text class="section" x="104" y="{top + 34}">{html.escape(heading)}</text>',
            f'<text class="legend" x="{canvas_width - 64}" y="{top + 32}" text-anchor="end" style="fill:#5f6d7c">{html.escape(detail)}</text>',
        ))
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


def _plotted_trajectory(result, label):
    evaluation = result["evaluations"][label]
    seed = result["trajectory_seed"]
    matches = [
        episode["trajectory"]
        for episode in evaluation["episodes"]
        if episode["seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"policy {label!r} must contain exactly one trajectory for seed {seed}"
        )
    return matches[0]


def _mapping_values(rows, name):
    values = []
    for row in rows:
        if not isinstance(row, Mapping) or name not in row:
            raise ValueError(f"trajectory mapping is missing required key {name!r}")
        values.append(float(row[name]))
    return _finite_array(f"trajectory mapping {name!r}", values)


def _series_panel(
    result,
    labels,
    *,
    title,
    field,
    index=None,
    reference_field=None,
    y_limits=None,
):
    series = []
    for label_index, label in enumerate(labels):
        trajectory = _plotted_trajectory(result, label)
        series.append(
            {
                "label": label,
                "color": _COLORS[label_index % len(_COLORS)],
                "time": trajectory["physical_time"],
                "values": trajectory[field],
                "index": index,
            }
        )
    reference = None
    if reference_field is not None:
        trajectory = _plotted_trajectory(result, labels[0])
        reference = {
            "time": trajectory["physical_time"],
            "values": trajectory[reference_field],
            "index": index,
        }
    return {
        "title": title,
        "series": series,
        "reference": reference,
        "y_limits": y_limits,
    }


def _single_series_panel(
    result,
    label,
    *,
    title,
    field,
    mapping_name,
    y_limits,
):
    trajectory = _plotted_trajectory(result, label)
    return {
        "title": title,
        "series": [
            {
                "label": mapping_name,
                "color": "#5f6368",
                "time": trajectory["physical_time"],
                "values": _mapping_values(trajectory[field], mapping_name),
                "index": None,
            }
        ],
        "reference": None,
        "y_limits": y_limits,
    }


def _disturbance_y_limits(values):
    array = _finite_array("disturbance values", values).reshape(-1)
    if np.all(np.isin(array, (0.0, 1.0))):
        return (0.0, 1.0)
    return None


def _draw_series_panel(panel, top, panel_index, *, left, right):
    chart_top = float(top + 36)
    bottom = float(top + 180)
    all_x = []
    all_y = []
    resolved_series = []
    for series in panel["series"]:
        time = _finite_array("trajectory time", series["time"])
        values = _series_component(series["values"], series["index"])
        _matching_lengths(time, values)
        all_x.extend(time.tolist())
        all_y.extend(values.tolist())
        resolved_series.append((series, time, values))
    reference = panel["reference"]
    resolved_reference = None
    if reference is not None:
        time = _finite_array("reference time", reference["time"])
        values = _series_component(reference["values"], reference["index"])
        _matching_lengths(time, values)
        all_x.extend(time.tolist())
        all_y.extend(values.tolist())
        resolved_reference = (time, values)
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
    time_unit = str(panel["time_unit"])

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
            f'<text class="tick" text-anchor="middle" x="{(left + right) / 2:.2f}" y="{bottom + 34}">time [{html.escape(time_unit)}]</text>',
            f'<g clip-path="url(#{clip_id})">',
        ]
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
    for series, time, values in resolved_series:
        color = series["color"]
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
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
    parts.append("</g>")
    return parts


def _series_component(raw_values, index):
    values = _finite_array("trajectory series", raw_values)
    if index is None:
        if values.ndim != 1:
            raise ValueError("scalar trajectory series must be one-dimensional")
        return values
    if values.ndim != 2 or index >= values.shape[1]:
        raise ValueError("vector trajectory series does not match its schema")
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
    return plot_range(array, include_zero=include_zero, padding_fraction=0.05)


def _map_x(value, minimum, maximum, left, right):
    return map_value(value, minimum, maximum, left, right)


def _map_y(value, minimum, maximum, top, bottom):
    return map_value(value, minimum, maximum, bottom, top)


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


def _number(value):
    number = float(value)
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude >= 1e4 or magnitude < 1e-3):
        return f"{number:.2e}"
    return f"{number:.3g}"


__all__ = ["render_trajectory_svg"]

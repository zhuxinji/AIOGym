"""SVG plotting helpers for benchmark artifacts."""
from __future__ import annotations

import math
from pathlib import Path

from aiogym.models import make_model
from .tables import TRACKING_COMPARISON_METRICS


_TRAJECTORY_ABSOLUTE_MIN_SPAN = 1e-6
_TRAJECTORY_RELATIVE_MIN_SPAN = 0.02
_TRAJECTORY_SCHEMA_MIN_SPAN = 0.02


def plot_summary(rows: list[dict], path: str, scenario: str):
    labels = [row["controller"] for row in rows]
    metrics = _summary_metrics(rows)
    colors = ["#3b82f6", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#64748b"]
    width = 1200
    panel_w, panel_h = 500, 250
    x_positions = (70, 650)
    nrows = max(1, (len(metrics) + 1) // 2)
    height = max(430, 95 + nrows * 335)
    panels = [
        (x_positions[i % 2], 95 + (i // 2) * 335, panel_w, panel_h)
        for i in range(len(metrics))
    ]
    title = f"{scenario} controller benchmark"
    parts = [_svg_header(width, height), _svg_text(60, 45, title, size=24, weight="700")]
    for panel, (key, panel_title) in zip(panels, metrics):
        x, y, w, h = panel
        vals = [float(row.get(key, 0.0)) for row in rows]
        parts.append(_svg_text(x, y - 22, panel_title, size=16, weight="700"))
        parts.extend(_svg_axes(x, y, w, h))
        if not vals:
            continue
        lo = min(min(vals), 0.0)
        hi = max(max(vals), 0.0)
        if hi == lo:
            hi = lo + 1.0
        zero = _map_y(0.0, lo, hi, y, h)
        parts.append(f'<line x1="{x}" y1="{zero:.2f}" x2="{x + w}" y2="{zero:.2f}" stroke="#94a3b8" stroke-dasharray="4 4"/>')
        gap = 18
        bar_w = max(10, (w - gap * (len(vals) + 1)) / max(len(vals), 1))
        for i, val in enumerate(vals):
            bx = x + gap + i * (bar_w + gap)
            by = _map_y(val, lo, hi, y, h)
            top, bottom = min(by, zero), max(by, zero)
            parts.append(
                f'<rect x="{bx:.2f}" y="{top:.2f}" width="{bar_w:.2f}" '
                f'height="{max(1.0, bottom - top):.2f}" fill="{colors[i % len(colors)]}"/>'
            )
            parts.append(_svg_text(bx + bar_w / 2, y + h + 24, labels[i], size=11, anchor="middle"))
            parts.append(_svg_text(bx + bar_w / 2, top - 6, _fmt(val), size=10, anchor="middle", fill="#334155"))
        parts.append(_svg_text(x - 12, y + 8, _fmt(hi), size=10, anchor="end", fill="#64748b"))
        parts.append(_svg_text(x - 12, y + h, _fmt(lo), size=10, anchor="end", fill="#64748b"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def _summary_metrics(rows: list[dict]) -> list[tuple[str, str]]:
    goal = str(rows[0].get("goal", "")) if rows else ""
    if goal == "regulation":
        return [
            ("tracking_cost", "Tracking Objective"),
            ("tracking_error_cost", "Tracking Error Cost"),
            ("tracking_move_cost", "Move Cost"),
        ]
    if goal == "economic":
        metrics = [
            ("profit", "Profit"),
            ("energy_kwh", "Energy kWh"),
            ("constraint", "Constraint penalty"),
        ]
        if _has_nonzero(rows, "production"):
            metrics.insert(1, ("production", "Production"))
        return metrics
    if goal == "safety":
        return [
            ("constraint_violation_count", "Constraint violations"),
            ("constraint_violation_severity", "Violation severity"),
            ("safety_margin_min", "Safety margin"),
        ]
    return [
        ("official_score", "Official score"),
        ("return", "Return"),
        ("energy_kwh", "Energy kWh"),
    ]


def _metric_label(metric: str) -> str:
    return "tracking cost" if metric == "tracking_cost" else metric


def _has_nonzero(rows: list[dict], key: str) -> bool:
    for row in rows:
        try:
            if abs(float(row.get(key, 0.0) or 0.0)) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def plot_rollouts(rollouts: list[dict], path: str, scenario: str):
    model = make_model(scenario)
    state_rows = list(model.state_schema())
    output_rows = list(model.controlled_output_schema())
    action_bounds = _combined_schema_bounds(model.action_schema())
    level_bounds = _quantity_schema_bounds(
        state_rows + output_rows, names=("level",), units=("m", "cm")
    )
    temperature_bounds = _quantity_schema_bounds(
        state_rows + output_rows,
        names=("temperature", "temp"),
        units=("degC", "K"),
    )
    if scenario == "cstr":
        panels = [
            {
                "title": "Concentration Ca",
                "series": [state_series(0)],
                "bounds": _combined_schema_bounds(state_rows[:1]),
            },
            {
                "title": "Temperature",
                "series": [
                    info_series("temps", 0),
                    setpoint_series("y_sp", 0, dashed=True),
                ],
                "bounds": temperature_bounds,
            },
            {
                "title": "Actuator commands",
                "series": action_series(model.action_names[:2]),
                "bounds": action_bounds,
            },
            {"title": "Stage profit", "series": [metric_series("profit")]},
            {"title": "Constraint penalty", "series": [metric_series("constraint")]},
        ]
    elif scenario == "hvac":
        panels = [
            {
                "title": "Zone temperatures",
                "series": info_vector_family("temps"),
                "bounds": temperature_bounds,
            },
            {
                "title": "Temperature setpoints",
                "series": setpoint_vector_family("y_sp"),
                "bounds": temperature_bounds,
            },
            {
                "title": "Actuator commands",
                "series": action_series(model.action_names),
                "bounds": action_bounds,
            },
            {"title": "Tracking error", "series": [metric_series("track")]},
            {"title": "Constraint penalty", "series": [metric_series("constraint")]},
        ]
    else:
        panels = [
            {
                "title": "Levels",
                "series": info_vector_family("levels"),
                "bounds": level_bounds,
            },
            {
                "title": "Temperatures",
                "series": info_vector_family("temps"),
                "bounds": temperature_bounds,
            },
            {
                "title": "Actuator commands",
                "series": action_series(model.action_names),
                "bounds": action_bounds,
            },
            {"title": "Tracking error", "series": [metric_series("track")]},
            {"title": "Constraint penalty", "series": [metric_series("constraint")]},
        ]
    _plot_series_panels(rollouts, panels, path, f"{scenario} rollout comparison")


def plot_tracking_control(rollouts: list[dict], path: str, scenario: str, case: str = "default"):
    """Plot state, setpoint, and actuator trajectories for one tracking case."""

    model = make_model(scenario)
    state_rows = list(model.state_schema())
    output_rows = list(model.controlled_output_schema())
    output_to_state = _controlled_state_indices(rollouts, state_rows, output_rows)
    state_to_output = {state_i: output_i for output_i, state_i in output_to_state.items()}
    controlled_state_indices = set(output_to_state.values())
    panels = []
    for i, row in enumerate(output_rows):
        if i in output_to_state:
            continue
        name = str(row.get("name") or f"y{i}")
        unit = str(row.get("unit") or "")
        panels.append({
            "title": _quantity_title(name, unit),
            "series": [
                info_series("y", i, label=name),
                info_series("y_sp", i, label=f"{name} setpoint", dashed=True, muted=True),
            ],
            "bounds": _combined_schema_bounds([row]),
        })
    for i, row in enumerate(state_rows):
        if i not in controlled_state_indices:
            continue
        name = str(row.get("name") or f"x{i}")
        unit = str(row.get("unit") or "")
        series = [state_series(i, label=name)]
        if i in state_to_output:
            series.append(info_series(
                "y_sp",
                state_to_output[i],
                label=f"{name} setpoint",
                dashed=True,
                muted=True,
            ))
        panels.append({
            "title": _quantity_title(name, unit),
            "series": series,
            "bounds": _combined_schema_bounds([row]),
        })
    action_scale = 1.0
    if scenario == "quadruple":
        action_scale = _quadruple_voltage_scale(rollouts, model)
    for i, row in enumerate(model.action_schema()):
        name = f"u{i + 1}" if scenario == "quadruple" else str(row.get("name") or f"u{i}")
        unit = "V" if scenario == "quadruple" else str(row.get("unit") or "")
        panels.append({
            "title": _quantity_title(name, unit),
            "series": [{"kind": "action", "index": i, "label": name, "scale": action_scale}],
            "bounds": _scaled_schema_bounds(row.get("bounds"), action_scale),
        })
    label = scenario if case == "default" else f"{scenario} / {case}"
    _plot_series_panels(rollouts, panels, path, f"{label} tracking control")


def _quadruple_voltage_scale(rollouts: list[dict], model) -> float:
    """Return the physical voltage represented by a normalized action of one."""

    for artifact in rollouts:
        params = (artifact.get("evaluation") or {}).get("model_params") or {}
        if params.get("max_voltage") is not None:
            return float(params["max_voltage"])
    return float(model.p["max_voltage"])


def _quantity_title(name: str, unit: str) -> str:
    return name if not unit else f"{name} ({unit})"


def _quantity_schema_bounds(rows, *, names: tuple[str, ...], units: tuple[str, ...]):
    selected = []
    normalized_units = {unit.lower() for unit in units}
    for row in rows:
        name = str(row.get("name") or "").lower()
        unit = str(row.get("unit") or "").lower()
        if any(token in name for token in names) or unit in normalized_units:
            selected.append(row)
    return _combined_schema_bounds(selected)


def _combined_schema_bounds(rows):
    bounds = [
        _scaled_schema_bounds(row.get("bounds"), 1.0)
        for row in rows
    ]
    bounds = [value for value in bounds if value is not None]
    if not bounds:
        return None
    return min(value[0] for value in bounds), max(value[1] for value in bounds)


def _scaled_schema_bounds(bounds, scale: float):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None
    try:
        lower, upper = float(bounds[0]) * scale, float(bounds[1]) * scale
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return None
    return lower, upper


def _controlled_state_indices(rollouts: list[dict], state_rows: list[dict],
                              output_rows: list[dict]) -> dict[int, int]:
    """Match controlled outputs to identical recorded state channels when possible."""

    matched = {}
    state_names = {str(row.get("name")): i for i, row in enumerate(state_rows)}
    for output_i, row in enumerate(output_rows):
        state_i = state_names.get(str(row.get("name")))
        if state_i is not None:
            matched[output_i] = state_i
    samples = [
        row
        for artifact in rollouts[:1]
        for row in artifact.get("rollout", [])[:32]
    ]
    used_states = set(matched.values())
    for output_i in range(len(output_rows)):
        if output_i in matched:
            continue
        for state_i in range(len(state_rows)):
            if state_i in used_states:
                continue
            pairs = [
                (
                    _list_value(row.get("info", {}).get("y"), output_i),
                    _list_value(row.get("next_state"), state_i),
                )
                for row in samples
            ]
            pairs = [(y, x) for y, x in pairs if y is not None and x is not None]
            if pairs and all(abs(float(y) - float(x)) <= 1e-9 for y, x in pairs):
                matched[output_i] = state_i
                used_states.add(state_i)
                break
    return matched


def plot_leaderboard(board: list[dict], path: str, title: str) -> None:
    metric = board[0]["metric"] if board else "metric"
    values = [0.0 if row["metric_value"] is None else float(row["metric_value"]) for row in board]
    width = 1080
    height = max(260, 130 + 54 * max(1, len(board)))
    left, top, bar_h = 220, 90, 26
    scale_w = 500
    value_x = left + scale_w + 118
    status_x = width - 110
    lo = min([0.0] + values)
    hi = max([0.0] + values)
    if lo == hi:
        hi = lo + 1.0
    parts = [_svg_header(width, height), _svg_text(42, 46, f"{title} leaderboard", size=22, weight="700")]
    parts.append(_svg_text(42, 74, f"Primary metric: {_metric_label(metric)}", size=12, fill="#475569"))
    parts.append(_svg_text(value_x, 74, "Value", size=11, anchor="end", fill="#64748b"))
    parts.append(_svg_text(status_x, 74, "Status", size=11, fill="#64748b"))
    for i, row in enumerate(board):
        y = top + i * 54
        value = 0.0 if row["metric_value"] is None else float(row["metric_value"])
        x0 = left + _map_x(min(0.0, value), lo, hi, 0, scale_w)
        x1 = left + _map_x(max(0.0, value), lo, hi, 0, scale_w)
        parts.append(_svg_text(42, y + 19, f"{row['rank']}. {row['controller']}", size=13, fill="#0f172a"))
        parts.append(f'<rect x="{min(x0, x1):.2f}" y="{y:.2f}" width="{max(2.0, abs(x1 - x0)):.2f}" height="{bar_h}" fill="#2563eb"/>')
        parts.append(_svg_text(value_x, y + 18, _fmt(value), size=12, anchor="end", fill="#334155"))
        parts.append(_svg_text(status_x, y + 18, str(row.get("execution_status") or ""), size=11, fill="#64748b"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def plot_grouped_leaderboard(sections: list[dict], path: str, title: str) -> None:
    width = 1180
    left, top = 250, 96
    scale_w = 520
    value_x = left + scale_w + 118
    status_x = width - 120
    section_gap = 28
    row_h = 36
    header_h = 46
    height = max(300, 120 + sum(header_h + row_h * max(1, len(section.get("board", []))) + section_gap for section in sections))
    parts = [_svg_header(width, height), _svg_text(42, 46, f"{title} leaderboard", size=22, weight="700")]
    y = top
    for section in sections:
        board = list(section.get("board") or [])
        section_title = str(section.get("title") or "benchmark")
        metric = str(section.get("metric") or (board[0]["metric"] if board else "metric"))
        values = [0.0 if row["metric_value"] is None else float(row["metric_value"]) for row in board]
        lo = min([0.0] + values)
        hi = max([0.0] + values)
        if lo == hi:
            hi = lo + 1.0
        parts.append(_svg_text(42, y, section_title, size=16, weight="700", fill="#0f172a"))
        parts.append(_svg_text(42, y + 22, f"Primary metric: {_metric_label(metric)}", size=11, fill="#64748b"))
        parts.append(_svg_text(value_x, y + 22, "Value", size=11, anchor="end", fill="#64748b"))
        parts.append(_svg_text(status_x, y + 22, "Status", size=11, fill="#64748b"))
        y += header_h
        for row in board:
            value = 0.0 if row["metric_value"] is None else float(row["metric_value"])
            x0 = left + _map_x(min(0.0, value), lo, hi, 0, scale_w)
            x1 = left + _map_x(max(0.0, value), lo, hi, 0, scale_w)
            parts.append(_svg_text(42, y + 19, f"{row['rank']}. {row['controller']}", size=13, fill="#0f172a"))
            parts.append(f'<rect x="{min(x0, x1):.2f}" y="{y:.2f}" width="{max(2.0, abs(x1 - x0)):.2f}" height="22" fill="#2563eb"/>')
            parts.append(_svg_text(value_x, y + 17, _fmt(value), size=12, anchor="end", fill="#334155"))
            parts.append(_svg_text(status_x, y + 17, str(row.get("execution_status") or ""), size=11, fill="#64748b"))
            y += row_h
        y += section_gap
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def plot_tracking_comparison_table(rows: list[dict], path: str, title: str) -> None:
    controllers = []
    for row in rows:
        for key in row:
            if key.endswith("_tracking_cost") and key != "best_tracking_cost":
                controllers.append(key[: -len("_tracking_cost")])
    controllers = list(dict.fromkeys(controllers))
    display_labels = {
        controller: "Oracle" if controller == "NMPC-oracle" else controller
        for controller in controllers
    }
    multiple_cases = len(rows) > 1
    table_rows = []
    for source in rows:
        case = str(source.get("case") or "default")
        for metric, label in TRACKING_COMPARISON_METRICS:
            values = {
                controller: _number_or_none(source.get(f"{controller}_{metric}"))
                for controller in controllers
            }
            ranked = [(controller, value) for controller, value in values.items() if value is not None]
            if not ranked:
                continue
            table_rows.append({
                "metric": f"{case} · {label}" if multiple_cases else label,
                "metric_key": metric,
                "best_controller": min(ranked, key=lambda item: item[1])[0],
                **values,
            })
    metric_width = 340 if multiple_cases else 240
    columns = [("metric", "Metric", metric_width)] + [
        (controller, display_labels[controller], 150)
        for controller in controllers
    ]
    left, top = 36, 76
    row_h, header_h = 36, 42
    table_width = sum(column[2] for column in columns)
    width = max(680, left * 2 + table_width)
    table_h = header_h + row_h * len(table_rows)
    height = max(260, top + table_h + 36)
    comparison_title = str(rows[0].get("case") or "default") if len(rows) == 1 else title
    parts = [
        _svg_header(width, height),
        _svg_text(
            36,
            46,
            f"{comparison_title} tracking comparison (mean per episode)",
            size=22,
            weight="700",
        ),
    ]
    _append_tracking_metric_table(
        parts, table_rows, columns, left, top, table_width, row_h, header_h
    )
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def _append_tracking_metric_table(
    parts: list[str],
    rows: list[dict],
    columns: list[tuple[str, str, int]],
    left: int,
    top: int,
    table_w: int,
    row_h: int,
    header_h: int,
) -> None:
    x = left
    parts.append(f'<rect x="{left}" y="{top}" width="{table_w}" height="{header_h}" fill="#e2e8f0"/>')
    for _, label, col_w in columns:
        parts.append(_svg_text(x + 8, top + 26, label, size=11, weight="700", fill="#334155"))
        x += col_w
    for r, row in enumerate(rows):
        y = top + header_h + r * row_h
        fill = "#ffffff" if r % 2 == 0 else "#f8fafc"
        parts.append(f'<rect x="{left}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}"/>')
        x = left
        best = str(row.get("best_controller") or "")
        for key, _, col_w in columns:
            value = row.get(key)
            text_fill = "#0f172a"
            weight = "400"
            if key == best:
                text_fill = "#047857"
                weight = "700"
            text = str(value) if key == "metric" else _table_value(value, str(row["metric_key"]))
            parts.append(_svg_text(x + 8, y + 23, text, size=11, fill=text_fill, weight=weight))
            x += col_w
    parts.append(f'<rect x="{left}" y="{top}" width="{table_w}" height="{header_h + row_h * len(rows)}" fill="none" stroke="#cbd5e1"/>')


def plot_constraint_timeline(rollouts: list[dict], path: str, scenario: str) -> None:
    width, height = 1100, 520
    left, top, w, h = 80, 82, 960, 330
    colors = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#475569"]
    series = []
    for artifact in rollouts:
        rows = artifact.get("rollout", [])
        if not rows:
            continue
        series.append({
            "name": artifact.get("controller_name", "controller"),
            "x": [float(row.get("time", i)) for i, row in enumerate(rows)],
            "y": [float(row.get("constraint", row.get("info", {}).get("constraint", 0.0)) or 0.0) for row in rows],
        })
    xs = [v for row in series for v in row["x"]]
    ys = [v for row in series for v in row["y"]]
    xlo, xhi = (min(xs), max(xs)) if xs else (0.0, 1.0)
    ylo, yhi = 0.0, max([1.0] + ys)
    if xlo == xhi:
        xhi = xlo + 1.0
    parts = [_svg_header(width, height), _svg_text(42, 46, f"{scenario} constraint timeline", size=22, weight="700")]
    parts.append(f'<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="#f8fafc" stroke="#cbd5e1"/>')
    if series and not any(abs(value) > 0.0 for value in ys):
        parts.append(_svg_text(
            left + w / 2,
            top + h / 2,
            "No constraint violations recorded",
            size=18,
            anchor="middle",
            fill="#047857",
            weight="700",
        ))
    for i, row in enumerate(series):
        points = []
        for xv, yv in zip(row["x"], row["y"]):
            px = left + (xv - xlo) / (xhi - xlo) * w
            py = top + h - (yv - ylo) / (yhi - ylo) * h
            points.append(f"{px:.2f},{py:.2f}")
        color = colors[i % len(colors)]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lx = left + (i % 4) * 210
        ly = top + h + 48 + (i // 4) * 24
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 30}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(_svg_text(lx + 38, ly + 4, row["name"], size=12, fill="#334155"))
    parts.append(_svg_text(left - 12, top + 10, _fmt(yhi), size=10, anchor="end", fill="#64748b"))
    _append_time_axis(parts, left, top, w, h, xlo, xhi, show_label=True)
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def plot_learning_curve(curve: list[dict], path: str, title: str) -> None:
    """Plot numeric RL training history rows as a compact SVG curve sheet."""

    curve = _deduplicate_learning_curve(curve)
    series_keys = _learning_curve_keys(curve)
    width, height = 1100, 640
    left, top, w, h = 86, 96, 920, 380
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#475569", "#0f766e", "#be123c"]
    parts = [_svg_header(width, height), _svg_text(42, 46, f"{title} learning curve", size=22, weight="700")]
    parts.extend(_svg_axes(left, top, w, h))
    if not curve or not series_keys:
        parts.append(_svg_text(left + w / 2, top + h / 2, "No learning-curve rows", size=15, anchor="middle", fill="#64748b"))
        parts.append("</svg>")
        _write_text(path, "\n".join(parts))
        return

    x_key = "timesteps" if any("timesteps" in row for row in curve) else "step"
    xs = [float(row.get(x_key, row.get("step", i))) for i, row in enumerate(curve)]
    xlo, xhi = min(0.0, min(xs)), max(xs)
    if xlo == xhi:
        xhi = xlo + 1.0
    all_ys = [float(row[key]) for row in curve for key in series_keys if _is_number(row.get(key))]
    ylo, yhi = _trajectory_axis_limits(all_ys)
    if all(key.startswith("tracking_") and key.endswith("cost") for key in series_keys):
        ylo = max(0.0, ylo)
    if series_keys == ["return"] and all(value <= 0 for value in all_ys):
        yhi = 0.0

    for i, key in enumerate(series_keys):
        ys = [float(row[key]) if _is_number(row.get(key)) else None for row in curve]
        clean_xs = [x for x, y in zip(xs, ys) if y is not None]
        clean_ys = [y for y in ys if y is not None]
        if not clean_xs:
            continue
        color = colors[i % len(colors)]
        points = _polyline_points(clean_xs, clean_ys, xlo, xhi, ylo, yhi, left, top, w, h)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lx = 42 + (i % 3) * 310
        ly = 530 + (i // 3) * 26
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        label = "reward" if key == "return" else _metric_label(key)
        parts.append(_svg_text(lx + 36, ly + 4, label, size=12, fill="#334155"))

    parts.append(_svg_text(left - 12, top + 8, _fmt(yhi), size=10, anchor="end", fill="#64748b"))
    parts.append(_svg_text(left - 12, top + h, _fmt(ylo), size=10, anchor="end", fill="#64748b"))
    parts.append(_svg_text(left, top + h + 30, _fmt(xlo), size=10, anchor="middle", fill="#64748b"))
    parts.append(_svg_text(left + w, top + h + 30, _fmt(xhi), size=10, anchor="middle", fill="#64748b"))
    parts.append(_svg_text(left + w / 2, top + h + 52, x_key, size=13, anchor="middle", fill="#334155"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def _deduplicate_learning_curve(curve: list[dict]) -> list[dict]:
    """Keep one row per timestep, preferring the final multi-episode result."""

    selected: dict[float, dict] = {}
    for index, row in enumerate(curve):
        raw_step = row.get("timesteps", row.get("step", index))
        step = float(raw_step) if _is_number(raw_step) else float(index)
        current = selected.get(step)
        if current is None or row.get("phase") == "final":
            selected[step] = row
    return [selected[step] for step in sorted(selected)]


def _learning_curve_keys(curve: list[dict]) -> list[str]:
    if any(_is_number(row.get("return")) for row in curve):
        return ["return"]

    tracking = [
        key for key in (
            "tracking_cost",
            "tracking_error_cost",
            "tracking_move_cost",
        )
        if any(_is_number(row.get(key)) for row in curve)
    ]
    if tracking:
        return tracking

    preferred = [
        "metric_value",
        "kpi",
        "profit",
        "return",
        "track",
        "tracking_iae",
        "constraint_violation_count",
        "constraint_violation_severity",
    ]
    skip = {"step", "timesteps", "time", "phase", "metric", "metric_direction"}
    keys = {
        key
        for row in curve
        for key, value in row.items()
        if key not in skip and _is_number(value)
    }
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys.difference(preferred)))
    return ordered[:8]


def state_series(index: int, label: str | None = None, dashed: bool = False):
    return {"kind": "state", "index": index, "label": label or f"x{index}", "dashed": dashed}


def info_series(key: str, index: int, label: str | None = None, dashed: bool = False,
                muted: bool = False):
    return {
        "kind": "info_vector",
        "key": key,
        "index": index,
        "label": label or f"{key}{index}",
        "dashed": dashed,
        "muted": muted,
    }


def setpoint_series(key: str, index: int, label: str | None = None, dashed: bool = False):
    return {"kind": "setpoint_vector", "key": key, "index": index, "label": label or f"{key}{index}", "dashed": dashed, "muted": dashed}


def metric_series(key: str, dashed: bool = False):
    return {"kind": "metric", "key": key, "label": key, "dashed": dashed}


def action_series(names):
    return [{"kind": "action", "index": i, "label": name} for i, name in enumerate(names)]


def info_vector_family(key: str):
    return [info_series(key, i) for i in range(_max_info_vector_len(key))]


def setpoint_vector_family(key: str):
    return [setpoint_series(key, i, dashed=True) for i in range(4)]


def _max_info_vector_len(key: str):
    return {"levels": 4, "temps": 4}.get(key, 1)


def _plot_series_panels(rollouts: list[dict], panels: list[dict], path: str, title: str):
    width = 1200
    left, right = 110, 40
    panel_h, gap = 160, 82
    top = 105
    height = max(400, top + len(panels) * (panel_h + gap) + 70)
    panel_boxes = [(left, top + i * (panel_h + gap), width - left - right, panel_h) for i in range(len(panels))]
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#475569", "#0f766e", "#be123c"]
    controller_names = list(dict.fromkeys(
        str(artifact.get("controller_name", "controller"))
        for artifact in rollouts
        if artifact.get("rollout")
    ))
    controller_colors = {
        name: colors[i % len(colors)]
        for i, name in enumerate(controller_names)
    }
    parts = [_svg_header(width, height), _svg_text(60, 48, title, size=28, weight="700")]

    for pidx, (panel, box) in enumerate(zip(panels, panel_boxes)):
        x, y, w, h = box
        parts.append(_svg_text(x, y - 24, panel["title"], size=20, weight="700"))
        parts.extend(_svg_axes(x, y, w, h))
        collected = []
        for artifact in rollouts:
            rows = artifact.get("rollout", [])
            if not rows:
                continue
            t = [float(row.get("time", i)) for i, row in enumerate(rows)]
            for spec in panel["series"]:
                ys = _extract_series(rows, spec)
                if any(v is not None for v in ys):
                    collected.append({
                        "controller": artifact.get("controller_name", "controller"),
                        "label": spec["label"],
                        "x": t,
                        "y": [0.0 if v is None else float(v) for v in ys],
                        "dashed": spec.get("dashed", False),
                        "muted": spec.get("muted", False),
                    })
        xs = [v for series in collected for v in series["x"]]
        ys = [v for series in collected for v in series["y"]]
        if not xs or not ys:
            continue
        xlo, xhi = min(xs), max(xs)
        ylo, yhi = _trajectory_axis_limits(ys, bounds=panel.get("bounds"))
        if xhi == xlo:
            xhi = xlo + 1.0
        for series in collected:
            color = controller_colors.get(series["controller"], colors[0])
            if series["muted"]:
                color = "#94a3b8"
            dash = ' stroke-dasharray="6 5"' if series["dashed"] else ""
            points = _polyline_points(series["x"], series["y"], xlo, xhi, ylo, yhi, x, y, w, h)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        parts.append(_svg_text(x - 14, y + 10, _fmt(yhi), size=15, anchor="end", fill="#475569"))
        parts.append(_svg_text(x - 14, y + h, _fmt(ylo), size=15, anchor="end", fill="#475569"))
        _append_time_axis(
            parts,
            x,
            y,
            w,
            h,
            xlo,
            xhi,
            show_label=pidx == len(panel_boxes) - 1,
        )

    legend_x, legend_y = 800, 32
    for i, name in enumerate(controller_names):
        lx = legend_x + (i % 3) * 125
        ly = legend_y + (i // 3) * 22
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{controller_colors[name]}" stroke-width="3"/>')
        parts.append(_svg_text(lx + 34, ly + 5, name, size=15, fill="#334155"))
    if any(spec.get("muted") for panel in panels for spec in panel["series"]):
        i = len(controller_names)
        lx = legend_x + (i % 3) * 125
        ly = legend_y + (i // 3) * 22
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="#94a3b8" stroke-width="3" stroke-dasharray="6 5"/>')
        parts.append(_svg_text(lx + 34, ly + 5, "Setpoint", size=15, fill="#334155"))
    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def _extract_series(rows: list[dict], spec: dict):
    if spec["kind"] == "state":
        return [_list_value(row.get("next_state"), spec["index"]) for row in rows]
    if spec["kind"] == "info_vector":
        return [_list_value(row.get("info", {}).get(spec["key"]), spec["index"]) for row in rows]
    if spec["kind"] == "setpoint_vector":
        return [_setpoint_value(row.get("setpoint", {}), spec["key"], spec["index"]) for row in rows]
    if spec["kind"] == "action":
        scale = float(spec.get("scale", 1.0))
        return [
            None if value is None else float(value) * scale
            for value in (_list_value(row.get("action"), spec["index"]) for row in rows)
        ]
    if spec["kind"] == "metric":
        return [row.get("info", {}).get(spec["key"]) for row in rows]
    raise ValueError(f"unknown plot series kind {spec['kind']!r}")


def _list_value(value, index: int):
    if isinstance(value, list) and len(value) > index:
        return value[index]
    return None


def _setpoint_value(setpoint: dict, key: str, index: int):
    return _list_value(setpoint.get(key), index)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _svg_header(width: int, height: int):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>'
    )


def _svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start",
              fill: str = "#0f172a", weight: str = "400"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}">{_escape(text)}</text>'
    )


def _svg_axes(x: float, y: float, w: float, h: float):
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#475569"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#475569"/>',
    ]


def _polyline_points(xs, ys, xlo, xhi, ylo, yhi, x, y, w, h):
    points = []
    for xv, yv in zip(xs, ys):
        px = x + (float(xv) - xlo) / (xhi - xlo) * w
        py = _map_y(float(yv), ylo, yhi, y, h)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _trajectory_axis_limits(values, *, bounds=None) -> tuple[float, float]:
    """Return padded limits that do not magnify floating-point jitter.

    Every process rollout, tracking-control figure, and learning curve uses this
    helper.  The minimum span is relative to the displayed operating point with
    a small absolute fallback for signals near zero.
    """

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return -0.5, 0.5
    data_lo, data_hi = min(finite), max(finite)
    reference = max(abs(data_lo), abs(data_hi))
    schema_bounds = _scaled_schema_bounds(bounds, 1.0)
    schema_minimum_span = (
        (schema_bounds[1] - schema_bounds[0]) * _TRAJECTORY_SCHEMA_MIN_SPAN
        if schema_bounds is not None
        else 0.0
    )
    minimum_span = max(
        _TRAJECTORY_ABSOLUTE_MIN_SPAN,
        reference * _TRAJECTORY_RELATIVE_MIN_SPAN,
        schema_minimum_span,
    )
    data_span = data_hi - data_lo
    display_span = max(data_span, minimum_span)
    center = 0.5 * (data_lo + data_hi)
    ylo = center - 0.5 * display_span
    yhi = center + 0.5 * display_span
    pad = display_span * 0.08
    ylo -= pad
    yhi += pad
    if schema_bounds is not None and data_lo >= schema_bounds[0] and data_hi <= schema_bounds[1]:
        if ylo < schema_bounds[0]:
            yhi += schema_bounds[0] - ylo
            ylo = schema_bounds[0]
        if yhi > schema_bounds[1]:
            ylo -= yhi - schema_bounds[1]
            yhi = schema_bounds[1]
    return ylo, yhi


def _append_time_axis(parts: list[str], x: float, y: float, w: float, h: float,
                      xlo: float, xhi: float, *, show_label: bool) -> None:
    for i in range(5):
        fraction = i / 4.0
        px = x + fraction * w
        value = xlo + fraction * (xhi - xlo)
        parts.append(
            f'<line x1="{px:.2f}" y1="{y + h:.2f}" x2="{px:.2f}" '
            f'y2="{y + h + 5:.2f}" stroke="#475569"/>'
        )
        parts.append(_svg_text(
            px,
            y + h + 23,
            _fmt_axis(value),
            size=15,
            anchor="middle",
            fill="#475569",
        ))
    if show_label:
        parts.append(_svg_text(
            x + w / 2,
            y + h + 50,
            "Time (s)",
            size=18,
            anchor="middle",
            fill="#334155",
        ))


def _fmt_axis(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _map_y(value: float, lo: float, hi: float, y: float, h: float):
    return y + h - (float(value) - lo) / (hi - lo) * h


def _map_x(value: float, lo: float, hi: float, x: float, w: float) -> float:
    return x + (float(value) - lo) / (hi - lo) * w


def _fmt(value: float):
    """Format plot values without rounding visible non-zero bars to zero."""

    value = float(value)
    if value == 0.0:
        return "0"
    return f"{value:.4g}"


def _table_value(value, key: str) -> str:
    number = _number_or_none(value)
    if number is None:
        return "" if value is None else str(value)
    if key in {"runtime_seconds", "runtime_total_seconds"} or key.endswith(
        ("_runtime_seconds", "_runtime_total_seconds")
    ):
        return f"{number:.2f}"
    return _fmt(number)


def _number_or_none(value):
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _escape(text: str):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_text(path: str, text: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)

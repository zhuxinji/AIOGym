"""Descriptively compare policies on the same ordered seeds."""

from __future__ import annotations

import html
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiogym.core.io import write_json

from .evaluate import evaluate


COMPARISON_SCHEMA_VERSION = "aiogym.comparison.v4"
TRAJECTORY_ARCHIVE_SCHEMA_VERSION = "aiogym.trajectory-archive.v1"
_TRAJECTORY_FIELDS = (
    "physical_time",
    "true_state",
    "output",
    "reference",
    "commanded_action",
    "channel_action",
    "applied_action",
    "reward",
    "disturbance",
    "constraint_costs",
    "minimum_safety_margin",
)
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
    """Compare policies and persist one compact report plus trajectory archive.

    Benchmark environments default to
    ``runs/<scenario>/benchmarks/<benchmark>`` and
    replace only the three managed comparison artifacts. Comparisons outside a
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
    first_schema = first["trajectory_schema"]
    static_schema_fields = (
        "time_unit",
        "state",
        "output",
        "action",
        "disturbance_names",
    )
    if any(
        any(
            result["trajectory_schema"][field] != first_schema[field]
            for field in static_schema_fields
        )
        for result in evaluations.values()
    ):
        raise ValueError("all policy evaluations must use compatible trajectory schemas")
    trajectory_schema = {
        **first_schema,
        "constraint_cost_names": sorted(
            {
                name
                for result in evaluations.values()
                for name in result["trajectory_schema"]["constraint_cost_names"]
            }
        ),
    }

    def ranking_key(label):
        result = evaluations[label]
        values = []
        for metric in ranking_metrics:
            median = result["aggregate"][metric["name"]]["median"]
            values.append(median if metric["direction"] == "minimize" else -median)
        return (*values, label)

    ordering = sorted(labels, key=ranking_key)
    trajectory_seed = ordered_seeds[0]
    archive = _write_trajectory_archive(
        output_directory / "trajectories.npz",
        evaluations,
        trajectory_schema,
        overwrite=overwrite,
    )
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "environment": dict(first["environment"]),
        "trajectory_schema": trajectory_schema,
        "seeds": list(ordered_seeds),
        "trajectory_seed": trajectory_seed,
        "trajectory_archive": archive,
        "max_steps": first["max_steps"],
        "ranking_metrics": ranking_metrics,
        "ordering": ordering,
        "evaluations": {
            label: _compact_evaluation(evaluations[label], trajectory_seed)
            for label in labels
        },
    }
    svg_path = output_directory / "comparison.svg"
    svg_path.write_text(render_trajectory_svg(result), encoding="utf-8")
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
    managed_names = {"comparison.json", "comparison.svg", "trajectories.npz"}
    invalid = sorted(
        path.name
        for path in entries
        if path.name in managed_names
        and (path.is_symlink() or not path.is_file())
    )
    if invalid:
        raise FileExistsError(
            f"default comparison artifacts must be files: {invalid}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory, overwrite


def _compact_evaluation(evaluation, trajectory_seed):
    episodes = []
    for episode in evaluation["episodes"]:
        compact_episode = {
            key: value for key, value in episode.items() if key != "trajectory"
        }
        if episode["seed"] == trajectory_seed:
            compact_episode["trajectory"] = episode["trajectory"]
        episodes.append(compact_episode)
    return {
        **{
            key: value
            for key, value in evaluation.items()
            if key not in {"episodes", "trajectory_summary"}
        },
        "episodes": episodes,
    }


def _write_trajectory_archive(path, evaluations, schema, *, overwrite):
    target = Path(path)
    arrays = {}
    entries = []
    columns = _trajectory_columns(schema)
    for policy_index, (label, evaluation) in enumerate(evaluations.items()):
        for episode_index, episode in enumerate(evaluation["episodes"]):
            prefix = f"p{policy_index:03d}_e{episode_index:03d}"
            trajectory_arrays = _trajectory_arrays(
                episode["trajectory"],
                columns,
            )
            for field, array in trajectory_arrays.items():
                arrays[f"{prefix}__{field}"] = array
            entries.append(
                {
                    "id": prefix,
                    "policy": label,
                    "seed": episode["seed"],
                    "length": episode["length"],
                }
            )
    _write_npz(target, arrays, overwrite=overwrite)
    return {
        "schema_version": TRAJECTORY_ARCHIVE_SCHEMA_VERSION,
        "file": target.name,
        "dtype": "float64",
        "fields": list(_TRAJECTORY_FIELDS),
        "columns": columns,
        "entries": entries,
    }


def _trajectory_columns(schema):
    state = [row["name"] for row in schema["state"]]
    output = [row["name"] for row in schema["output"]]
    action = [row["name"] for row in schema["action"]]
    return {
        "true_state": state,
        "output": output,
        "reference": output,
        "commanded_action": action,
        "channel_action": action,
        "applied_action": action,
        "disturbance": list(schema["disturbance_names"]),
        "constraint_costs": list(schema["constraint_cost_names"]),
    }


def _trajectory_arrays(trajectory, columns):
    missing = set(_TRAJECTORY_FIELDS) - set(trajectory)
    if missing:
        raise ValueError(f"trajectory fields are missing: {sorted(missing)}")
    lengths = {field: len(trajectory[field]) for field in _TRAJECTORY_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"trajectory fields have inconsistent lengths: {lengths}")
    length = next(iter(lengths.values()))
    arrays = {}
    scalar_fields = {
        "physical_time",
        "reward",
        "minimum_safety_margin",
    }
    mapping_fields = {"disturbance", "constraint_costs"}
    for field in _TRAJECTORY_FIELDS:
        if field in scalar_fields:
            array = np.asarray(trajectory[field], dtype=np.float64)
            expected_shape = (length,)
        elif field in mapping_fields:
            names = columns[field]
            rows = trajectory[field]
            unexpected = sorted(
                {
                    name
                    for row in rows
                    for name in row
                    if name not in names
                }
            )
            if unexpected:
                raise ValueError(
                    f"trajectory field {field!r} contains unknown columns: {unexpected}"
                )
            if field == "disturbance":
                missing_names = sorted(
                    {
                        name
                        for row in rows
                        for name in names
                        if name not in row
                    }
                )
                if missing_names:
                    raise ValueError(
                        f"trajectory field {field!r} is missing columns: {missing_names}"
                    )
            array = np.asarray(
                [
                    [float(row[name]) if name in row else 0.0 for name in names]
                    for row in rows
                ],
                dtype=np.float64,
            ).reshape(length, len(names))
            expected_shape = (length, len(names))
        else:
            array = np.asarray(trajectory[field], dtype=np.float64)
            expected_shape = (length, len(columns[field]))
        if array.shape != expected_shape:
            raise ValueError(
                f"trajectory field {field!r} must have shape {expected_shape}; "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"trajectory field {field!r} must contain finite values")
        arrays[field] = array
    return arrays


def _write_npz(path, arrays, *, overwrite):
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_output_directory(env) -> Path:
    base_env = env.unwrapped
    if base_env.benchmark is None:
        raise ValueError(
            "output is required when comparing policies outside a benchmark"
        )
    scenario_id = base_env.scenario.id.replace("_", "-")
    benchmark_id = base_env.benchmark.id
    return Path("runs") / scenario_id / "benchmarks" / benchmark_id


def render_trajectory_svg(result, *, title: str | None = None) -> str:
    labels = tuple(result["evaluations"])
    schema = result["trajectory_schema"]
    time_unit = str(schema["time_unit"])
    resolved_title = (
        f"{result['environment']['scenario']} policy comparison"
        if title is None
        else str(title)
    )
    temperature_indices = tuple(
        index
        for index, row in enumerate(schema["output"])
        if row.get("unit") == "degC"
    )
    temperature_limits = None
    if temperature_indices:
        temperature_values = []
        for label in labels:
            trajectory = _plotted_trajectory(result, label)
            output = _finite_array("trajectory output", trajectory["output"])
            for index in temperature_indices:
                temperature_values.extend(output[:, index].tolist())
        reference = _finite_array(
            "trajectory reference",
            _plotted_trajectory(result, labels[0])["reference"],
        )
        for index in temperature_indices:
            temperature_values.extend(reference[:, index].tolist())
        temperature_limits = _plot_range(
            temperature_values,
            include_zero=False,
        )
    output_panels = []
    for index, row in enumerate(schema["output"]):
        if "level" in str(row["name"]).lower():
            y_limits = _schema_bounds(row)
        elif index in temperature_indices:
            y_limits = temperature_limits
        else:
            y_limits = None
        output_panels.append(
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
    action_panels = []
    for index, row in enumerate(schema["action"]):
        action_panels.append(
            _series_panel(
                result,
                labels,
                title=f"Applied action: {_schema_label(row)}",
                field="applied_action",
                index=index,
                y_limits=_schema_bounds(row),
            )
        )
    action_panels.append(
        _boundary_distance_panel(result, labels, schema, time_unit=time_unit)
    )
    disturbance_panels = []
    first_trajectory = _plotted_trajectory(result, labels[0])
    for name in schema["disturbance_names"]:
        values = _mapping_values(
            first_trajectory["disturbance"],
            name,
        )
        if values.size and float(np.max(values) - np.min(values)) > 1e-12:
            disturbance_panels.append(
                _single_series_panel(
                    result,
                    labels[0],
                    title=f"Disturbance: {name}",
                    field="disturbance",
                    mapping_name=name,
                )
            )

    for panel in (*output_panels, *action_panels, *disturbance_panels):
        panel["time_unit"] = time_unit

    width = 1440
    header_height = 100
    panel_height = 225
    return_panel_height = max(285, 100 + 34 * len(labels))
    panel_rows = max(len(output_panels), len(action_panels))
    compact_hydraulic_layout = (
        not disturbance_panels
        and result["environment"]["scenario"] == "three_tank"
        and result["environment"]["benchmark"]
        in {"tracking", "boundary-safety"}
        and len(output_panels) == 3
        and len(action_panels) == 5
    )
    compact_single_output_layout = (
        not disturbance_panels
        and len(output_panels) == 1
        and len(action_panels) == 3
    )
    if disturbance_panels:
        output_panels.sort(
            key=lambda panel: "level" not in panel["title"].lower()
        )
        compact_groups = (disturbance_panels, output_panels, action_panels)
        compact_rows = sum(
            math.ceil(len(panels) / min(3, len(panels)))
            for panels in compact_groups
        )
        height = header_height + panel_height * compact_rows + return_panel_height
    elif compact_hydraulic_layout or compact_single_output_layout:
        height = header_height + 3 * panel_height + return_panel_height
    else:
        height = header_height + panel_height * panel_rows + return_panel_height
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
            f"{html.escape(resolved_title)}"
            "</text>"
        ),
        (
            f'<text class="tick" x="64" y="62">trajectory seed: '
            f"{html.escape(str(result['trajectory_seed']))}; return seeds: "
            f"{html.escape(', '.join(str(seed) for seed in result['seeds']))}"
            "</text>"
        ),
    ]
    legend_x = 720
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
    if disturbance_panels:
        group_top = header_height
        panel_index = 0
        grid_left = 76.0
        grid_right = float(width - 40)
        grid_gap = 48.0
        for panels in compact_groups:
            columns = min(3, len(panels))
            panel_width = (
                grid_right - grid_left - grid_gap * (columns - 1)
            ) / columns
            for index, panel in enumerate(panels):
                row, column = divmod(index, columns)
                left = grid_left + column * (panel_width + grid_gap)
                right = left + panel_width
                parts.extend(
                    _draw_series_panel(
                        panel,
                        group_top + row * panel_height,
                        panel_index,
                        left=left,
                        right=right,
                    )
                )
                panel_index += 1
            group_top += math.ceil(len(panels) / columns) * panel_height
        return_top = group_top
    elif compact_hydraulic_layout:
        grid_left = 76.0
        grid_right = float(width - 40)
        grid_gap = 48.0
        panel_index = 0
        output_width = (
            grid_right - grid_left - 2 * grid_gap
        ) / len(output_panels)
        for column, panel in enumerate(output_panels):
            left = grid_left + column * (output_width + grid_gap)
            parts.extend(
                _draw_series_panel(
                    panel,
                    header_height,
                    panel_index,
                    left=left,
                    right=left + output_width,
                )
            )
            panel_index += 1
        actuator_panels = action_panels[:-1]
        action_width = (
            grid_right - grid_left - 3 * grid_gap
        ) / len(actuator_panels)
        for column, panel in enumerate(actuator_panels):
            left = grid_left + column * (action_width + grid_gap)
            parts.extend(
                _draw_series_panel(
                    panel,
                    header_height + panel_height,
                    panel_index,
                    left=left,
                    right=left + action_width,
                )
            )
            panel_index += 1
        parts.extend(
            _draw_series_panel(
                action_panels[-1],
                header_height + 2 * panel_height,
                panel_index,
                left=grid_left,
                right=grid_right,
            )
        )
        return_top = header_height + 3 * panel_height
    elif compact_single_output_layout:
        grid_left = 76.0
        grid_right = float(width - 40)
        grid_gap = 48.0
        panel_index = 0
        parts.extend(
            _draw_series_panel(
                output_panels[0],
                header_height,
                panel_index,
                left=grid_left,
                right=grid_right,
            )
        )
        panel_index += 1
        actuator_panels = action_panels[:-1]
        action_width = (
            grid_right - grid_left - grid_gap
        ) / len(actuator_panels)
        for column, panel in enumerate(actuator_panels):
            left = grid_left + column * (action_width + grid_gap)
            parts.extend(
                _draw_series_panel(
                    panel,
                    header_height + panel_height,
                    panel_index,
                    left=left,
                    right=left + action_width,
                )
            )
            panel_index += 1
        parts.extend(
            _draw_series_panel(
                action_panels[-1],
                header_height + 2 * panel_height,
                panel_index,
                left=grid_left,
                right=grid_right,
            )
        )
        return_top = header_height + 3 * panel_height
    else:
        for column, panels in enumerate((output_panels, action_panels)):
            for row, panel in enumerate(panels):
                top = header_height + row * panel_height
                left = 76.0 + column * 720.0
                right = 680.0 + column * 720.0
                panel_index = row * 2 + column
                parts.extend(
                    _draw_series_panel(
                        panel,
                        top,
                        panel_index,
                        left=left,
                        right=right,
                    )
                )
        return_top = header_height + panel_rows * panel_height
    parts.extend(_draw_return_distribution(result, labels, return_top, width=width))
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
    horizontal=None,
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
        "horizontal": horizontal,
        "y_limits": y_limits,
        "annotations": [],
    }


def _boundary_distance_panel(result, labels, schema, *, time_unit):
    level_states = []
    for index, row in enumerate(schema["state"]):
        name = str(row["name"])
        lower_name = name.lower()
        is_level = "level" in lower_name or (
            lower_name.startswith("h") and lower_name[1:].isdigit()
        )
        if is_level:
            bounds = _schema_bounds(row)
            if bounds is None:
                raise ValueError("level state schema must declare finite bounds")
            low, high = bounds
            if "unit" not in row:
                raise ValueError("level state schema must declare a physical unit")
            level_states.append((index, name, str(row["unit"]), low, high, 1.0))
    if level_states:
        units = {
            unit for _index, _name, unit, _low, _high, _scale in level_states
        }
        if len(units) != 1:
            raise ValueError("level state schema must use one physical unit")
        selected_states = level_states
        unit = next(iter(units))
        title = f"Closest level-boundary distance [{unit}]"
    else:
        selected_states = []
        for index, row in enumerate(schema["state"]):
            bounds = _schema_bounds(row)
            if bounds is None:
                continue
            low, high = bounds
            selected_states.append(
                (
                    index,
                    str(row["name"]),
                    "fraction",
                    low,
                    high,
                    1.0 / (high - low),
                )
            )
        if not selected_states:
            raise ValueError("trajectory schema must contain bounded states")
        unit = "fraction"
        title = "Closest state-boundary distance [fraction]"

    series = []
    annotations = []
    all_closest = []
    for label_index, label in enumerate(labels):
        trajectory = _plotted_trajectory(result, label)
        time = _finite_array("trajectory time", trajectory["physical_time"])
        state = _finite_array("trajectory true_state", trajectory["true_state"])
        if state.ndim != 2 or state.shape[1] != len(schema["state"]):
            raise ValueError("trajectory true_state does not match its schema")
        _matching_lengths(time, state)
        distances = []
        constraints = []
        for state_index, name, _unit, low, high, scale in selected_states:
            distances.extend(
                (
                    (state[:, state_index] - low) * scale,
                    (high - state[:, state_index]) * scale,
                )
            )
            tank = _level_state_label(name)
            constraints.extend((f"{tank} lower", f"{tank} upper"))
        distance_matrix = np.column_stack(distances)
        closest = np.min(distance_matrix, axis=1)
        all_closest.extend(closest.tolist())
        time_index, constraint_index = np.unravel_index(
            int(np.argmin(distance_matrix)),
            distance_matrix.shape,
        )
        color = _COLORS[label_index % len(_COLORS)]
        series.append(
            {
                "label": label,
                "color": color,
                "time": time,
                "values": closest,
                "index": None,
                "marker": {
                    "time": float(time[time_index]),
                    "value": float(closest[time_index]),
                    "label": constraints[constraint_index],
                },
            }
        )
        annotations.append(
            {
                "color": color,
                "text": (
                    f"{label}: {constraints[constraint_index]}, "
                    f"{_number(closest[time_index])} {unit} @ "
                    f"{_number(time[time_index])} {time_unit}"
                ),
            }
        )
    minimum = float(np.min(all_closest))
    maximum = float(np.max(all_closest))
    y_limits = (
        (0.0, 1.0 if maximum == 0.0 else 1.05 * maximum)
        if minimum >= 0.0
        else _plot_range(all_closest, include_zero=True)
    )
    return {
        "title": title,
        "series": series,
        "reference": None,
        "horizontal": 0.0,
        "y_limits": y_limits,
        "annotations": annotations,
    }


def _level_state_label(name):
    lower = str(name).lower()
    if lower.startswith("h") and lower[1:].isdigit():
        return f"tank {int(lower[1:])}"
    return str(name).replace("_", " ")


def _single_series_panel(
    result,
    label,
    *,
    title,
    field,
    mapping_name,
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
        "horizontal": None,
        "y_limits": None,
        "annotations": [],
    }


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
        marker = series.get("marker")
        if marker is not None:
            marker_x = _map_x(marker["time"], x_min, x_max, left, right)
            marker_y = _map_y(marker["value"], y_min, y_max, chart_top, bottom)
            parts.append(
                f'<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="4.5" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1.5">'
                f'<title>{html.escape(series["label"])}: '
                f'{html.escape(marker["label"])}, {_number(marker["value"])} '
                f'@ {_number(marker["time"])} {html.escape(time_unit)}</title></circle>'
            )
    parts.append("</g>")
    annotations = panel["annotations"]
    if annotations:
        line_height = 14.0
        box_width = min(330.0, 0.58 * (right - left))
        box_height = 8.0 + line_height * len(annotations)
        box_left = right - box_width
        parts.append(
            f'<rect x="{box_left:.2f}" y="{chart_top + 4:.2f}" '
            f'width="{box_width:.2f}" height="{box_height:.2f}" rx="4" '
            'fill="#ffffff" fill-opacity="0.88" stroke="#dadce0"/>'
        )
        for index, annotation in enumerate(annotations):
            y = chart_top + 17.0 + index * line_height
            parts.append(
                f'<text class="tick" text-anchor="end" x="{right - 7:.2f}" '
                f'y="{y:.2f}" style="fill:{annotation["color"]}">'
                f'{html.escape(annotation["text"])}</text>'
            )
    return parts


def _draw_return_distribution(result, labels, top, *, width):
    left = 160.0
    right = float(width - 40)
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
    if any(len(values) != len(result["seeds"]) for values in distributions.values()):
        raise ValueError("return distributions must match the comparison seeds")
    all_values = np.concatenate(tuple(distributions.values()))
    value_min, value_max = _plot_range(all_values.tolist(), include_zero=False)
    parts = [
        f'<text class="panel-title" x="76" y="{top + 22}">Cumulative return by policy</text>',
        f'<text class="tick" x="720" y="{top + 22}">circles: all return seeds; diamond: median</text>',
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
        for seed, value, offset in zip(result["seeds"], values, offsets):
            x = _map_x(value, value_min, value_max, left, right)
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y + offset:.2f}" r="4" '
                f'fill="{color}" fill-opacity="0.48">'
                f'<title>seed {seed}: {_number(value)}</title></circle>'
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


def _number(value):
    number = float(value)
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude >= 1e4 or magnitude < 1e-3):
        return f"{number:.2e}"
    return f"{number:.3g}"


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "compare_policies",
    "render_trajectory_svg",
]

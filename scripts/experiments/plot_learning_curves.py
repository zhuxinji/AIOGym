"""Render validation learning curves as a dependency-free SVG."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def plot_learning_curves(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    source_path = Path(source)
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"learning-curve plot already exists: {output_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") if "runs" in payload else [payload]
    series = []
    for run in runs:
        artifact_dir = _reference_path(
            run.get("artifact_dir"), source_path.parent
        )
        benchmark_path = artifact_dir / "benchmark.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        panels = _run_panels(benchmark)
        if panels:
            series.append((str(run.get("training_seed")), panels))
    if not series:
        raise ValueError("no learning-curve points found")
    svg = _render_svg(series)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return {
        "source": str(source_path),
        "output": str(output_path),
        "series": len(series),
        "points": sum(
            len(panels[0][1]) for _, panels in series if panels
        ),
        "panels": [label for label, _ in series[0][1]],
    }


def _run_panels(benchmark: dict) -> list[tuple[str, list[tuple[int, float]]]]:
    curve = list(benchmark.get("learning_curve") or ())
    case_ids = _case_ids(benchmark, curve)
    aggregate = []
    case_points: list[list[tuple[int, float]]] = []
    for row in curve:
        step = int(row.get("timesteps", row.get("step", -1)))
        value = row.get("metric_value")
        if step < 0 or value is None:
            continue
        aggregate.append((step, float(value)))
        values = row.get("case_values")
        if isinstance(values, list):
            while len(case_points) < len(values):
                case_points.append([])
            for index, case_value in enumerate(values):
                if case_value is not None:
                    case_points[index].append((step, float(case_value)))
    if not aggregate:
        return []
    metric = next(
        (str(row.get("metric")) for row in curve if row.get("metric")),
        "validation metric",
    )
    panels = [(f"aggregate {metric}", aggregate)]
    panels.extend(
        (
            case_ids[index] if index < len(case_ids) else f"Case {index + 1}",
            points,
        )
        for index, points in enumerate(case_points)
        if points
    )
    return panels


def _case_ids(benchmark: dict, curve: list[dict]) -> list[str]:
    for row in curve:
        values = row.get("case_ids")
        if isinstance(values, list) and values:
            return [str(value) for value in values]
    evaluation = benchmark.get("track_evaluation") or {}
    return [
        str(result["case_id"])
        for result in evaluation.get("results") or ()
        if result.get("case_id")
    ]


def _render_svg(series) -> str:
    panel_labels = [label for label, _ in series[0][1]]
    width = 1000
    left, right, top, bottom = 105, 35, 65, 65
    panel_height, panel_gap = 190, 70
    height = top + len(panel_labels) * (panel_height + panel_gap) + bottom
    all_points = [
        point
        for _, panels in series
        for _, points in panels
        for point in points
    ]
    x_values = [point[0] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_max = x_min + 1

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")
    body = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="40" y="35" font-size="22" font-weight="700">Validation learning curves by Case</text>',
    ]
    for panel_index, label in enumerate(panel_labels):
        panel_top = top + panel_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        panel_series = [
            (seed, dict(panels).get(label, [])) for seed, panels in series
        ]
        y_values = [value for _, points in panel_series for _, value in points]
        use_log = bool(y_values) and all(value > 0.0 for value in y_values)
        transformed = [math.log10(value) for value in y_values] if use_log else y_values
        y_min, y_max = min(transformed), max(transformed)
        if y_min == y_max:
            y_max = y_min + 1.0

        def xy(
            point,
            *,
            log_scale=use_log,
            plot_bottom=panel_bottom,
            plot_y_min=y_min,
            plot_y_max=y_max,
        ):
            x = left + (point[0] - x_min) / (x_max - x_min) * (
                width - left - right
            )
            value = math.log10(point[1]) if log_scale else point[1]
            y = plot_bottom - (
                (value - plot_y_min) / (plot_y_max - plot_y_min) * panel_height
            )
            return x, y

        scale = "log scale" if use_log else "linear scale"
        body.extend([
            f'<text x="{left}" y="{panel_top-16}" font-size="15" font-weight="700">{html.escape(label)} ({scale})</text>',
            f'<line x1="{left}" y1="{panel_bottom}" x2="{width-right}" y2="{panel_bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_bottom}" stroke="#111827"/>',
            f'<text x="{left-10}" y="{panel_top+5}" text-anchor="end">{(10**y_max if use_log else y_max):.6g}</text>',
            f'<text x="{left-10}" y="{panel_bottom}" text-anchor="end">{(10**y_min if use_log else y_min):.6g}</text>',
            f'<text x="{left}" y="{panel_bottom+22}">{x_min}</text>',
            f'<text x="{width-right}" y="{panel_bottom+22}" text-anchor="end">{x_max}</text>',
        ])
        for index, (_seed, points) in enumerate(panel_series):
            if not points:
                continue
            color = colors[index % len(colors)]
            coordinates = " ".join(
                f"{x:.2f},{y:.2f}" for x, y in map(xy, points)
            )
            body.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for x, y in map(xy, points):
                body.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="white" '
                    f'stroke="{color}" stroke-width="2"/>'
                )
    legend_y = height - 25
    for index, (seed, _) in enumerate(series):
        color = colors[index % len(colors)]
        body.append(
            f'<text x="{left + index * 150}" y="{legend_y}" fill="{color}">seed {html.escape(seed)}</text>'
        )
    body.append(
        f'<text x="{width/2}" y="{height-48}" text-anchor="middle">Environment transitions; markers are validation checkpoints</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _reference_path(value, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = plot_learning_curves(
        args.source,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

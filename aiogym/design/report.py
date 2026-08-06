"""Dependency-free JSON, HTML, and SVG design-study reports."""
from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path

from aiogym._internal.serialization import write_json_artifact

from .study import DESIGN_RESULT_SCHEMA_VERSION, DESIGN_SWEEP_SCHEMA_VERSION


def render_design_report(result: dict) -> tuple[str, str]:
    """Render an existing design result or sweep into HTML and SVG."""

    schema = result.get("schema_version")
    if schema == DESIGN_RESULT_SCHEMA_VERSION:
        return _render_single(result)
    if schema == DESIGN_SWEEP_SCHEMA_VERSION:
        return _render_sweep(result)
    raise ValueError(f"unsupported design report schema: {schema!r}")


def write_design_report_bundle(
    result: dict,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write report.json, report.html, and summary.svg without silent replacement."""

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    targets = {
        "json": directory / "report.json",
        "html": directory / "report.html",
        "svg": directory / "summary.svg",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "design report target already exists: " + ", ".join(str(path) for path in existing)
        )
    html_text, svg_text = render_design_report(result)
    write_json_artifact(targets["json"], result, overwrite=overwrite)
    _atomic_text(targets["html"], html_text, overwrite=overwrite)
    _atomic_text(targets["svg"], svg_text, overwrite=overwrite)
    return {name: str(path) for name, path in targets.items()}


def load_design_result(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"design result not found: {path}")
    with source.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("schema_version") not in {
        DESIGN_RESULT_SCHEMA_VERSION,
        DESIGN_SWEEP_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported design report schema: {result.get('schema_version')!r}")
    return result


def _render_single(result):
    verdict = result["verdict"]
    colour = "#15803d" if verdict == "PASS" else "#b91c1c"
    gates = result["gates"]
    steady = result["steady_state"]
    dynamic = result["dynamic"]
    robustness = result["robustness"]
    gate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(gate['name'])}</td>"
        f"<td class={'pass' if gate['passed'] else 'fail'}>"
        f"{'PASS' if gate['passed'] else 'FAIL'}</td>"
        f"<td>{html.escape('; '.join(gate['reasons']))}</td>"
        "</tr>"
        for gate in gates
    )
    action_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{_number(value)}</td>"
        "</tr>"
        for name, value in zip(steady["action_names"], steady["action"])
    )
    limitations = "".join(
        f"<li>{html.escape(item)}</li>" for item in result["limitations"]
    )
    svg = _single_svg(result, colour)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIO-Gym design report: {html.escape(result['design_id'])}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#172033;background:#fff}}
h1{{margin-bottom:.25rem}} .verdict{{font-size:1.5rem;font-weight:700;color:{colour}}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #d7dce5;padding:.55rem;text-align:left}}
th{{background:#f3f5f8}} .pass{{color:#15803d;font-weight:700}} .fail{{color:#b91c1c;font-weight:700}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem}}
.card{{border:1px solid #d7dce5;border-radius:8px;padding:.8rem}} code{{word-break:break-all}}
</style></head><body>
<h1>{html.escape(result['design_id'])}</h1>
<div class="verdict">{verdict}</div>
<p>Evidence: {html.escape(result['evidence_status'])}</p>
<code>{html.escape(result['design_hash'])}</code>
<h2>Decision gates</h2><table><thead><tr><th>Gate</th><th>Status</th><th>Reasons</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Key metrics</h2><div class="metrics">
<div class="card">Steady actuator margin<br><strong>{_number(steady['actuator_margin'])}</strong></div>
<div class="card">Heat-up time<br><strong>{_number(dynamic['heatup_time_s'])} s</strong></div>
<div class="card">Dynamic energy<br><strong>{_number(dynamic['energy_kwh'])} kWh</strong></div>
<div class="card">Robustness pass rate<br><strong>{_number(robustness['pass_rate'])}</strong></div>
</div>
<h2>Steady actions</h2><table><thead><tr><th>Actuator</th><th>Command</th></tr></thead><tbody>{action_rows}</tbody></table>
<h2>Visual summary</h2>{svg}
<h2>Limitations</h2><ul>{limitations}</ul>
</body></html>"""
    return page, svg


def _render_sweep(result):
    candidates = result["candidates"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(candidate['design_id'])}</td>"
        f"<td>{_number(candidate['value'])}</td>"
        f"<td class={'pass' if candidate['verdict'] == 'PASS' else 'fail'}>{candidate['verdict']}</td>"
        f"<td>{_number(candidate['steady_max_action'])}</td>"
        f"<td>{_number(candidate['heatup_time_s'])}</td>"
        f"<td>{_number(candidate['energy_kwh'])}</td>"
        f"<td>{_number(candidate['robustness_pass_rate'])}</td>"
        "</tr>"
        for candidate in candidates
    )
    svg = _sweep_svg(result)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIO-Gym design sweep</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033;background:#fff}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7dce5;padding:.5rem;text-align:left}}th{{background:#f3f5f8}}
.pass{{color:#15803d;font-weight:700}}.fail{{color:#b91c1c;font-weight:700}}
</style></head><body><h1>Design sweep</h1>
<p>Base design: {html.escape(result['base_design_id'])}<br>Parameter: <code>{html.escape(result['parameter'])}</code></p>
<table><thead><tr><th>Design</th><th>Value</th><th>Verdict</th><th>Steady max action</th><th>Heat-up s</th><th>Energy kWh</th><th>Robust pass</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Visual summary</h2>{svg}</body></html>"""
    return page, svg


def _single_svg(result, colour):
    steady = result["steady_state"]
    names = steady["action_names"]
    actions = steady["action"]
    thermal_duties = [None, None, None, *steady.get("heater_to_liquid_power_w", [])]
    required_limit = max(0.0, min(1.0, float(steady["required_maximum_action"])))
    width = 1120
    row_height = 48
    height = 106 + row_height * len(names)
    plot_x = 190
    plot_width = 470
    bar_height = 22
    bars = []
    for index, (name, raw) in enumerate(zip(names, actions)):
        raw_value = None if raw is None else float(raw)
        value = 0.0 if raw_value is None else max(0.0, min(1.0, raw_value))
        thermal_duty = thermal_duties[index] if index < len(thermal_duties) else None
        if raw_value is None:
            fill = "#64748b"
            assessment = "not installed"
        elif thermal_duty is not None and float(thermal_duty) < -1e-6:
            fill = "#7c3aed"
            assessment = (
                f"{100 * raw_value:.1f}% heat; "
                f"{abs(float(thermal_duty)) / 1000:.2f} kW cooling needed - NO COOLER"
            )
        elif raw_value > 1.0:
            fill = "#b91c1c"
            assessment = (
                f"{100 * raw_value:.1f}% required - NEEDS {raw_value:.2f}x CAPACITY"
            )
        elif raw_value > required_limit:
            fill = "#d97706"
            assessment = f"{100 * raw_value:.1f}% required - LOW RESERVE"
        else:
            fill = "#15803d"
            assessment = f"{100 * raw_value:.1f}% required - OK"
        y = 88 + index * row_height
        limit_x = plot_x + plot_width * required_limit
        capacity_x = plot_x + plot_width
        overflow_marker = ""
        if raw_value is not None and raw_value > 1.0:
            overflow_marker = (
                f'<polygon points="{capacity_x + 3},{y + 2} '
                f'{capacity_x + 18},{y + bar_height / 2} '
                f'{capacity_x + 3},{y + bar_height - 2}" fill="#b91c1c"/>'
            )
        bars.append(
            f'<text x="20" y="{y + 16}" font-size="14" font-weight="600">'
            f'{html.escape(name)}</text>'
            f'<rect x="{plot_x}" y="{y}" width="{plot_width}" height="{bar_height}" '
            f'fill="#e2e8f0" rx="4"/>'
            f'<rect x="{plot_x}" y="{y}" width="{plot_width * value:.2f}" '
            f'height="{bar_height}" fill="{fill}" rx="4"/>'
            f'<line x1="{limit_x:.2f}" y1="{y - 3}" x2="{limit_x:.2f}" '
            f'y2="{y + bar_height + 3}" stroke="#d97706" stroke-width="2" '
            f'stroke-dasharray="3 2"/>'
            f'<line x1="{capacity_x}" y1="{y - 3}" x2="{capacity_x}" '
            f'y2="{y + bar_height + 3}" stroke="#334155" stroke-width="2"/>'
            f'{overflow_marker}'
            f'<text x="700" y="{y + 16}" font-size="13" fill="{fill}">'
            f'{html.escape(assessment)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto" '
        f'role="img" aria-label="Required steady-state actuator command as a percentage of full command">'
        f'<rect width="{width}" height="{height}" fill="#fff"/>'
        f'<g fill="#172033">'
        f'<text x="20" y="28" font-size="20" font-weight="700">Steady-state actuator demand</text>'
        f'<text x="20" y="51" font-size="13">How hard each actuator must work at the target operating point; '
        f'orange dashed line = {100 * required_limit:.0f}% design limit, dark line = 100% full command</text>'
        f'<text x="{width - 20}" y="28" text-anchor="end" font-size="15" '
        f'font-weight="700" fill="{colour}">Overall design: {html.escape(result["verdict"])}</text>'
        f'<text x="20" y="76" font-size="12" font-weight="600">Actuator</text>'
        f'<text x="{plot_x}" y="76" font-size="12">0%</text>'
        f'<text x="{plot_x + plot_width * required_limit - 7:.2f}" y="76" font-size="12" '
        f'text-anchor="end" fill="#a16207">{100 * required_limit:.0f}%</text>'
        f'<text x="{plot_x + plot_width + 7}" y="76" font-size="12">100%</text>'
        f'<text x="700" y="76" font-size="12" font-weight="600">Required demand / conclusion</text>'
        f'{"".join(bars)}</g></svg>'
    )


def _sweep_svg(result):
    candidates = result["candidates"]
    width = 800
    height = 330
    left = 55
    bottom = 270
    plot_height = 220
    count = max(1, len(candidates))
    bar_width = min(80, 600 / count * 0.65)
    gap = 650 / count
    elements = [
        f'<line x1="{left}" y1="{bottom}" x2="760" y2="{bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="50" x2="{left}" y2="{bottom}" stroke="#334155"/>',
        '<text x="8" y="45" font-size="12">pass rate</text>',
    ]
    for index, candidate in enumerate(candidates):
        rate = float(candidate["robustness_pass_rate"])
        x = left + 30 + index * gap
        y = bottom - plot_height * rate
        colour = "#15803d" if candidate["verdict"] == "PASS" else "#b91c1c"
        elements.extend(
            (
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{plot_height * rate:.2f}" fill="{colour}"/>',
                f'<text x="{x:.2f}" y="{bottom + 18}" font-size="11">{_number(candidate["value"])}</text>',
                f'<text x="{x:.2f}" y="{max(45, y - 5):.2f}" font-size="11">{rate:.2f}</text>',
            )
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto" '
        f'role="img" aria-label="Design sweep robustness pass rates">'
        f'<rect width="{width}" height="{height}" fill="#fff"/>'
        f'<g fill="#172033">{"".join(elements)}</g></svg>'
    )


def _atomic_text(path: Path, text: str, *, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"report target already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


__all__ = [
    "load_design_result",
    "render_design_report",
    "write_design_report_bundle",
]

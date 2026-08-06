from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiogym.design import (
    render_design_report,
    run_design_study,
    run_design_sweep,
    write_design_report_bundle,
)


EXAMPLE = (
    Path(__file__).parents[2]
    / "configs"
    / "design"
    / "cascade-recirculating-example-v1.json"
)


def _raw_example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_complete_design_study_separates_all_decision_gates():
    result = run_design_study(EXAMPLE, robustness_samples=2, seed=7)

    assert result["verdict"] == "PASS"
    assert [gate["name"] for gate in result["gates"]] == [
        "static_engineering",
        "model_readiness",
        "steady_state_feasibility",
        "hardware_interlocks",
        "dynamic_commissioning",
        "robustness",
    ]
    assert all(gate["passed"] for gate in result["gates"])
    assert result["steady_state"]["maximum_derivative_residual"] < 1e-8
    assert result["dynamic"]["heatup_time_s"] is not None
    assert result["robustness"]["samples"] == 2


def test_inadequate_heater_design_fails_without_nonfinite_json_values():
    spec = _raw_example()
    spec["heaters"] = []
    spec["uncertainties"]["samples"] = 0
    result = run_design_study(spec)

    assert result["verdict"] == "FAIL"
    assert not result["steady_state"]["passed"]
    assert result["steady_state"]["action"][3] is None
    json.dumps(result, allow_nan=False)


def test_visual_summary_explains_required_command_without_clipping_overcapacity():
    spec = _raw_example()
    spec["heaters"][0]["power_w"] = 100.0
    spec["uncertainties"]["samples"] = 0
    result = run_design_study(spec)

    _, svg = render_design_report(result)

    required = float(result["steady_state"]["action"][3])
    assert required > 1.0
    assert "100% full command" in svg
    assert "Required demand / conclusion" in svg
    assert f"NEEDS {required:.2f}x CAPACITY" in svg


def test_report_bundle_is_self_contained_and_no_overwrite(tmp_path):
    result = run_design_study(EXAMPLE, robustness_samples=0)
    paths = write_design_report_bundle(result, tmp_path)

    assert set(paths) == {"json", "html", "svg"}
    assert json.loads((tmp_path / "report.json").read_text())["design_hash"] == result[
        "design_hash"
    ]
    assert "simulation-screening-only" in (tmp_path / "report.html").read_text()
    html_report = (tmp_path / "report.html").read_text()
    svg_summary = (tmp_path / "summary.svg").read_text()
    assert "background:#fff" in html_report
    assert svg_summary.startswith("<svg")
    assert 'fill="#fff"' in svg_summary
    assert 'fill="#172033"' in svg_summary
    with pytest.raises(FileExistsError):
        write_design_report_bundle(result, tmp_path)


def test_numeric_sweep_keeps_candidate_identities_and_comparison_metrics():
    result = run_design_sweep(
        EXAMPLE,
        parameter="pump.max_flow_m3s",
        values=[0.00035, 0.00045],
        robustness_samples=0,
    )

    assert result["schema_version"] == "aiogym.design_sweep.v1"
    assert len(result["candidates"]) == 2
    assert len({candidate["design_hash"] for candidate in result["candidates"]}) == 2
    assert all("steady_max_action" in candidate for candidate in result["candidates"])

from __future__ import annotations

import json

import pytest

from aiogym.cli.main import main
from aiogym.design import load_design_spec, prompt_design_spec


def test_wizard_defaults_produce_valid_si_design():
    messages = []
    spec = prompt_design_spec(
        "wizard-default",
        input_fn=lambda _prompt: "",
        output_fn=messages.append,
    )

    assert spec["id"] == "wizard-default"
    assert spec["tanks"][0]["area_m2"] == pytest.approx(0.075)
    assert spec["tanks"][0]["height_m"] == pytest.approx(0.4)
    assert spec["pump"]["motor_power_w"] == pytest.approx(370.0)
    assert spec["pump"]["max_flow_m3s"] * 60000.0 == pytest.approx(25.0)
    assert spec["operation"]["circulation_flow_m3s"] * 60000.0 == pytest.approx(5.0)
    assert len(spec["heaters"]) == 3
    assert messages[-1].startswith("Design inputs validated")


def test_wizard_converts_custom_engineering_units():
    answers = iter(
        [
            "400,300,500",
            "",
            "",
            "1:2.5:80",
            "0.55,30,2,12",
            "6",
            "250,240,240",
            "",
            "",
            "60",
            "3",
            "90",
            "5",
        ]
    )
    spec = prompt_design_spec(
        "wizard-custom",
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _message: None,
    )

    assert spec["tanks"][0]["area_m2"] == pytest.approx(0.12)
    assert spec["tanks"][0]["height_m"] == pytest.approx(0.5)
    assert spec["heaters"] == [
        {"tank": 1, "power_w": 2500.0, "efficiency": 0.8}
    ]
    assert spec["pump"]["motor_power_w"] == pytest.approx(550.0)
    assert spec["pump"]["max_flow_m3s"] * 60000.0 == pytest.approx(30.0)
    assert spec["operation"]["circulation_flow_m3s"] * 60000.0 == pytest.approx(6.0)
    assert spec["requirements"]["maximum_heatup_time_s"] == pytest.approx(3600.0)
    assert spec["requirements"]["robustness_pass_rate"] == pytest.approx(0.9)
    assert spec["uncertainties"]["samples"] == 5


def test_advanced_wizard_defaults_also_validate():
    spec = prompt_design_spec(
        "wizard-advanced",
        advanced=True,
        input_fn=lambda _prompt: "",
        output_fn=lambda _message: None,
    )

    assert load_design_spec(spec)["id"] == "wizard-advanced"


def test_design_new_writes_spec_without_manual_json_editing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    output = tmp_path / "wizard.json"

    assert main(["design", "new", "cli-wizard", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["id"] == "cli-wizard"
    assert payload["schema_version"] == "aiogym.design_spec.v1"


def test_design_run_interactive_can_save_and_execute(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    spec_path = tmp_path / "saved.json"
    report_dir = tmp_path / "report"

    assert main(
        [
            "design",
            "run",
            "--interactive",
            "--name",
            "interactive-run",
            "--save-spec",
            str(spec_path),
            "--robustness-samples",
            "0",
            "--output",
            str(report_dir),
        ]
    ) == 0

    assert spec_path.is_file()
    result = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert result["design_id"] == "interactive-run"
    assert result["verdict"] == "PASS"


@pytest.mark.parametrize("advanced", (False, True))
def test_bare_design_command_creates_runs_and_reports(
    advanced,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aiogym.cli.design._default_design_name",
        lambda: "automatic-design",
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "0" if "Robustness samples" in prompt else "",
    )
    command = ["design", "--advanced"] if advanced else ["design"]

    assert main(command) == 0

    spec = json.loads(
        (tmp_path / "designs" / "automatic-design.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (tmp_path / "runs" / "design" / "automatic-design" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["id"] == "automatic-design"
    assert report["design_id"] == "automatic-design"
    assert report["verdict"] == "PASS"

from __future__ import annotations

import json
from pathlib import Path

from aiogym.cli.main import main


EXAMPLE = (
    Path(__file__).parents[2]
    / "configs"
    / "design"
    / "cascade-recirculating-example-v1.json"
)


def test_design_validate_and_run_commands(tmp_path, capsys):
    assert main(["design", "validate", str(EXAMPLE), "--json"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["schema_version"] == "aiogym.design_spec.v1"

    output = tmp_path / "study"
    assert main(
        [
            "design",
            "run",
            str(EXAMPLE),
            "--robustness-samples",
            "0",
            "--output",
            str(output),
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["verdict"] == "PASS"
    assert (output / "report.json").is_file()
    assert (output / "report.html").is_file()
    assert (output / "summary.svg").is_file()

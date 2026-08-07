from __future__ import annotations

import json

import pytest

from aiogym.cli.main import main


def test_cli_lists_only_v2_resources(capsys):
    assert main(["list", "tasks", "--scenario", "three_tank"]) == 0
    assert capsys.readouterr().out == (
        "three_tank/economic\nthree_tank/regulation\n"
    )
    assert main(["list", "plants", "--scenario", "three_tank"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "lab-three-tank-v1",
        "open-cascade-v1",
        "recirculating-h1-v1",
    ]
    assert main(
        [
            "list",
            "conditions",
            "--scenario",
            "three_tank",
            "--plant",
            "lab-three-tank-v1",
        ]
    ) == 0
    assert capsys.readouterr().out == "commissioning\n"


def test_cli_design_new_and_validate(tmp_path, capsys):
    plant = tmp_path / "plant.json"
    assert main(["design", "new", "three_tank", str(plant)]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["schema_version"] == "aiogym.plant.v2"
    assert plant.is_file()
    assert main(["design", "validate", str(plant)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["plant"]["plant_hash"]


def test_cli_collects_dataset_v4(tmp_path, capsys):
    output = tmp_path / "dataset"
    assert (
        main(
            [
                "collect",
                "three_tank/regulation",
                "--controller",
                "hold",
                "--episodes",
                "1",
                "--max-steps",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["manifest"]["schema_version"] == "aiogym.dataset.v4"
    assert result["transitions"] == 2


def test_removed_command_has_migration_hint():
    with pytest.raises(SystemExit, match="migration-v0.2"):
        main(["benchmark"])

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiogym.core import make_env
from aiogym.workflows.design import (
    convert_design_spec_v1,
    load_plant,
    study,
    sweep,
    validate_plant,
)


DESIGN = (
    Path(__file__).resolve().parents[2]
    / "aiogym/scenarios/three_tank/default-design-v1.json"
)


def _raw_design():
    return json.loads(DESIGN.read_text(encoding="utf-8"))


def test_design_v1_converts_to_generic_plant_and_shared_env_hash():
    converted = convert_design_spec_v1(DESIGN)
    assert converted["schema_version"] == "aiogym.plant.v2"
    assert converted["scenario"] == "three_tank"
    plant = load_plant(converted)
    assert load_plant(DESIGN).plant_hash == plant.plant_hash
    resolved = validate_plant(plant)
    env = make_env("three_tank/regulation", plant=plant, condition="commissioning")
    try:
        _, info = env.reset(seed=0)
        assert resolved.plant_hash == plant.plant_hash == info["plant_hash"]
        assert env.action_space.shape == (6,)
    finally:
        env.close()


def test_design_study_preserves_gates_and_uses_common_dynamic_path():
    result = study(DESIGN, robustness_samples=0, seed=7)
    assert result["verdict"] == "PASS"
    assert [row["category"] for row in result["checks"]] == [
        "static",
        "model",
        "steady_state",
        "safety",
        "dynamic",
        "robustness",
    ]
    assert result["dynamic"]["rollout_executor"] == "aiogym.core.rollout"
    assert result["dynamic"]["controller"]["kind"] == "matrix_pid"
    assert result["dynamic"]["heatup_time_s"] == pytest.approx(890.0)
    assert result["dynamic"]["energy_kwh"] == pytest.approx(
        1.6565875831380923, rel=5e-6
    )
    assert result["dynamic"]["final_temperature_error_degC"] == pytest.approx(
        0.3250202855539186, rel=1e-5
    )


def test_inadequate_heater_fails_and_result_is_strict_json():
    plant = convert_design_spec_v1(_raw_design())
    plant["plant"]["heaters"][0]["power_w"] = 100.0
    result = study(plant, robustness_samples=0, seed=0)
    assert result["verdict"] == "FAIL"
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_sweep_changes_plant_hash_and_artifacts_are_no_overwrite(tmp_path):
    result = sweep(
        DESIGN,
        parameter="plant.heaters.0.power_w",
        values=(1900.0, 2100.0),
        robustness_samples=0,
        seed=3,
    )
    assert len({row["plant_hash"] for row in result["candidates"]}) == 2
    output = tmp_path / "study"
    written = study(
        DESIGN,
        robustness_samples=0,
        seed=3,
        output=output,
    )
    assert set(Path(path).name for path in written["artifacts"].values()) == {
        "manifest.json",
        "report.json",
        "report.md",
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["plant_hash"] == written["plant_hash"]
    assert manifest["task_hash"] == written["task_hash"]
    assert manifest["condition_hash"] == written["condition_hash"]
    assert manifest["interface_hash"] == written["interface_hash"]
    assert manifest["env_hash"] == written["env_hash"]
    with pytest.raises(FileExistsError):
        study(
            DESIGN,
            robustness_samples=0,
            seed=3,
            output=output,
        )


def test_robustness_sampling_is_seeded_and_reports_same_samples():
    first = study(DESIGN, robustness_samples=2, seed=7)
    second = study(DESIGN, robustness_samples=2, seed=7)
    assert first["robustness"]["pass_rate"] == 1.0
    assert first["robustness"]["cases"] == second["robustness"]["cases"]
    assert all(np.isfinite(row["dynamic"]["energy_kwh"]) for row in first["robustness"]["cases"])
    assert all(
        row["dynamic"]["disturbance"] == row["disturbance"]
        for row in first["robustness"]["cases"]
    )
    assert [
        row["dynamic"]["heatup_time_s"] for row in first["robustness"]["cases"]
    ] != [890.0, 890.0]


@pytest.mark.parametrize(
    "plant_id", ("open-cascade-v1", "recirculating-h1-v1")
)
def test_design_study_dispatches_both_supported_topologies(plant_id):
    result = study(plant_id, robustness_samples=0, seed=0)
    assert result["verdict"] == "PASS"
    assert result["condition_id"] in {
        "continuous-benchmark",
        "commissioning",
    }
    assert len(result["study_hash"]) == 64
    assert len(result["interface_hash"]) == 64
    assert len(result["env_hash"]) == 64

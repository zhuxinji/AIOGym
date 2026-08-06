from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiogym.design import compile_design_model


EXAMPLE = (
    Path(__file__).parents[2]
    / "configs"
    / "design"
    / "cascade-recirculating-example-v1.json"
)


def _raw_example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_design_model_has_stable_six_action_contract_and_equilibrium():
    model = compile_design_model(EXAMPLE)
    equilibrium = model.nominal_steady_state()

    assert model.action_names == (
        "pump_P101",
        "valve_V12",
        "valve_V23",
        "heater_H1",
        "heater_H2",
        "heater_H3",
    )
    assert model.action_dim() == 6
    assert model.heater_mask == (True, True, True)
    assert equilibrium["feasible"]
    assert max(
        abs(value)
        for value in model.dynamics(
            equilibrium["state"],
            equilibrium["action"],
            model.disturbance_defaults(),
        )
    ) < 1e-10


def test_absent_heaters_remain_masked_at_full_command():
    spec = _raw_example()
    spec["heaters"] = [spec["heaters"][0]]
    model = compile_design_model(spec)
    state = model.initial_state()
    levels = [state[0], state[2], state[4]]
    temperatures = [state[1], state[3], state[5]]
    info = model.process_info(
        state, levels, temperatures, {}, [1.0] * model.action_dim()
    )

    assert model.heater_mask == (True, False, False)
    assert info["H1_electric_power_w"] == pytest.approx(2000.0)
    assert info["H2_electric_power_w"] == 0.0
    assert info["H3_electric_power_w"] == 0.0


def test_design_model_independent_mass_and_energy_balances_close():
    model = compile_design_model(EXAMPLE)
    residuals = model.balance_residuals(
        [0.20, 25.0, 0.25, 24.0, 0.27, 23.0],
        [0.5, 0.25, 0.3, 0.4, 0.2, 0.1],
        {
            "t_amb": 18.0,
            "pump_flow_factor": 0.9,
            "heater_efficiency": 0.85,
            "heat_loss_factor": 1.2,
        },
    )

    assert max(abs(value) for value in residuals["tank_mass_balance_m3s"]) < 1e-15
    assert abs(residuals["total_mass_balance_m3s"]) < 1e-15
    assert max(abs(value) for value in residuals["tank_energy_balance_w"]) < 1e-8
    assert abs(residuals["total_energy_balance_w"]) < 1e-8

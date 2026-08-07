from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aiogym.scenarios  # noqa: F401 - registers built-in scenarios
from aiogym.core import make_env


ca = pytest.importorskip("casadi")
pytestmark = pytest.mark.symbolic

PLANT_PATH = (
    Path(__file__).resolve().parents[2]
    / "aiogym/scenarios/three_tank/plants/lab-three-tank-v1.json"
)


def _plant(*heater_tanks):
    raw = json.loads(PLANT_PATH.read_text(encoding="utf-8"))
    raw["id"] = "symbolic-layout-" + "-".join(str(tank) for tank in heater_tanks)
    raw["plant"]["heaters"] = [
        row for row in raw["plant"]["heaters"] if row["tank"] in heater_tanks
    ]
    return raw


@pytest.mark.parametrize("heater_tanks", ((1,), (1, 3), (1, 2, 3)))
def test_parameterized_layout_has_matching_casadi_dynamics_and_energy(heater_tanks):
    env = make_env("three_tank/regulation", plant=_plant(*heater_tanks))
    try:
        model = env.model._model
        state = ca.MX.sym("state", 6)
        action = ca.MX.sym("action", env.action_space.shape[0])
        dynamics = model.dynamics(
            state,
            action,
            model.disturbance_vector(dict(env.condition.disturbances)),
            backend="casadi",
            ca=ca,
        )
        energy = model.energy_kw(action, backend="casadi", ca=ca)
        function = ca.Function("plant", [state, action], [dynamics, energy])

        numeric_state = np.asarray(env.condition.initial_state, dtype=float)
        numeric_action = np.linspace(0.2, 0.8, env.action_space.shape[0])
        symbolic_dx, symbolic_energy = function(numeric_state, numeric_action)
        numeric_dx = model.dynamics(
            numeric_state,
            numeric_action,
            dict(env.condition.disturbances),
        )

        assert np.asarray(symbolic_dx).reshape(-1) == pytest.approx(numeric_dx)
        assert float(symbolic_energy) == pytest.approx(model.energy_kw(numeric_action))
    finally:
        env.close()

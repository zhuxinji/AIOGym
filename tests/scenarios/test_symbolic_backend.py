from __future__ import annotations

import numpy as np
import pytest

from aiogym.core.env import make_env


ca = pytest.importorskip("casadi")
pytestmark = pytest.mark.symbolic


def test_bom_model_has_matching_casadi_dynamics_and_energy():
    env = make_env("three_tank", reward="regulation")
    try:
        model = env.model
        state = ca.MX.sym("state", 6)
        action = ca.MX.sym("action", env.action_space.shape[0])
        dynamics = model.dynamics(
            state,
            action,
            model.disturbance_vector(dict(env.episode.disturbances)),
            backend="casadi",
            ca=ca,
        )
        energy = model.energy_kw(action, backend="casadi", ca=ca)
        function = ca.Function("scenario_model", [state, action], [dynamics, energy])

        numeric_state = np.asarray(env.episode.initial_state, dtype=float)
        numeric_action = np.linspace(0.2, 0.8, env.action_space.shape[0])
        symbolic_dx, symbolic_energy = function(numeric_state, numeric_action)
        numeric_dx = model.dynamics(
            numeric_state,
            numeric_action,
            dict(env.episode.disturbances),
        )

        assert np.asarray(symbolic_dx).reshape(-1) == pytest.approx(numeric_dx)
        assert float(symbolic_energy) == pytest.approx(model.energy_kw(numeric_action))
    finally:
        env.close()

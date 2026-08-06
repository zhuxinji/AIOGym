from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aiogym
from aiogym.controllers import make_controller


GOLDEN = json.loads(
    (
        Path(__file__).with_name("golden")
        / "three-tank-unification-phase0-v1.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("plant_id", tuple(GOLDEN["cases"]))
def test_three_tank_unification_phase0_fixture_is_reproducible(plant_id):
    expected = GOLDEN["cases"][plant_id]
    env = aiogym.make_env(
        expected["legacy_task"], preset=expected["legacy_preset"]
    )
    try:
        observation, info = env.reset(seed=GOLDEN["seed"])
        model = getattr(env.model, "_model", env.model)
        assert [row["name"] for row in env.model.state_schema()] == expected[
            "state_schema"
        ]
        assert [row["name"] for row in env.model.action_schema()] == expected[
            "action_schema"
        ]
        assert [row["name"] for row in model.setpoint_schema()] == expected[
            "output_schema"
        ]
        assert observation == pytest.approx(expected["initial_observation"], abs=1e-8)
        assert env.model.default_action() == pytest.approx(
            expected["default_action"], abs=1e-12
        )

        for controller_id, contract in expected["controller_actions"].items():
            if contract["status"] == "unsupported":
                assert contract["error_type"] == "ValueError"
                continue
            policy = make_controller(
                controller_id,
                env=env,
                profile=expected["legacy_preset"],
            )
            policy.reset(seed=GOLDEN["seed"])
            assert policy.act(observation, {"info": info}) == pytest.approx(
                contract["action"], abs=1e-8
            )

        action = np.asarray(expected["fixed_action"], dtype=np.float32)
        for row in expected["steps"]:
            observation, reward, terminated, truncated, info = env.step(action)
            state = np.asarray(info["true_state"], dtype=float)
            energy_kw = float(
                info.get(
                    "energy_kw",
                    env.model.action_energy_kw(
                        action, state, info["disturbance"]
                    ),
                )
            )
            assert state == pytest.approx(row["state"], abs=1e-12)
            assert observation == pytest.approx(row["observation"], abs=1e-8)
            assert reward == pytest.approx(row["reward"], abs=1e-12)
            assert energy_kw == pytest.approx(row["energy_kw"], abs=1e-12)
            assert info["constraint_costs"] == row["constraints"]
            assert list(info.get("protection_events", ())) == row[
                "protection_events"
            ]
            assert not terminated
            assert not truncated
    finally:
        env.close()

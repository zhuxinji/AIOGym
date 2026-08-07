from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aiogym.controllers import make_controller
from aiogym.core import make_env, rollout
from aiogym.core.backends import _NUMERIC_OPS
from aiogym.scenarios.three_tank.actuators import ActuatorLayout
from aiogym.workflows import collect


PLANT_PATH = (
    Path(__file__).resolve().parents[2]
    / "aiogym/scenarios/three_tank/plants/lab-three-tank-v1.json"
)


def _plant(*heater_tanks):
    raw = json.loads(PLANT_PATH.read_text(encoding="utf-8"))
    raw["id"] = "layout-" + "-".join(str(tank) for tank in heater_tanks)
    raw["plant"]["heaters"] = [
        row for row in raw["plant"]["heaters"] if row["tank"] in heater_tanks
    ]
    return raw


@pytest.mark.parametrize(
    ("heater_tanks", "names"),
    [
        (
            (1,),
            ("pump_P101", "valve_V12", "valve_V23", "heater_H1"),
        ),
        (
            (1, 3),
            (
                "pump_P101",
                "valve_V12",
                "valve_V23",
                "heater_H1",
                "heater_H3",
            ),
        ),
        (
            (1, 2, 3),
            (
                "pump_P101",
                "valve_V12",
                "valve_V23",
                "heater_H1",
                "heater_H2",
                "heater_H3",
            ),
        ),
    ],
)
def test_public_actions_contain_only_installed_equipment(heater_tanks, names):
    env = make_env("three_tank/regulation", plant=_plant(*heater_tanks))
    try:
        assert env.action_space.shape == (len(names),)
        assert tuple(row["name"] for row in env.model.action_schema()) == names
        assert len(env.model.default_action()) == len(names)
    finally:
        env.close()


def test_actuator_layout_round_trip_and_physical_equivalence():
    env = make_env("three_tank/regulation", plant=_plant(1, 3))
    try:
        model = env.model._model
        layout = model.action_layout
        public = [0.4, 0.5, 0.6, 0.7, 0.8]
        internal = layout.expand(public)
        assert internal == [0.4, 0.5, 0.6, 0.7, 0.0, 0.8]
        assert layout.compress(internal) == public

        state = list(env.condition.initial_state)
        disturbances = dict(env.condition.disturbances)
        public_dx = model.dynamics(state, public, disturbances)
        internal_dx = model._dynamics(
            model.state_vector(state),
            internal,
            disturbances,
            _NUMERIC_OPS,
        )
        assert public_dx == pytest.approx(internal_dx, abs=1e-14)
        expected_energy = (
            public[0] ** 3 * model.p["pump_power_max"]
            + public[3] * model.p["heater_power"][0]
            + public[4] * model.p["heater_power"][2]
        ) / 1000.0
        assert model.energy_kw(public) == pytest.approx(expected_energy)
    finally:
        env.close()


def test_layout_validation_rejects_duplicate_and_zero_power_heaters():
    duplicate = _plant(1, 2, 3)
    duplicate["plant"]["heaters"].append(
        copy.deepcopy(duplicate["plant"]["heaters"][0])
    )
    with pytest.raises(ValueError, match="declared more than once"):
        make_env("three_tank/regulation", plant=duplicate)

    zero_power = _plant(1)
    zero_power["plant"]["heaters"][0]["power_w"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        make_env("three_tank/regulation", plant=zero_power)


def test_layout_changes_interface_and_controller_dimensions():
    interfaces = set()
    plants = set()
    for tanks, expected_dim in (((1,), 4), ((1, 3), 5), ((1, 2, 3), 6)):
        env = make_env("three_tank/regulation", plant=_plant(*tanks))
        try:
            interfaces.add(env.identity.interface_hash)
            plants.add(env.identity.plant_hash)
            observation, info = env.reset(seed=2)
            pid = make_controller("pid", env=env)
            assert pid.act(observation, {"reference": info["reference"]}).shape == (
                expected_dim,
            )
            mpc = make_controller("mpc", env=env)
            assert mpc.controller.nu == expected_dim
            result = rollout(env, mpc, seed=2, max_steps=2)
            assert all(row.action.shape == (expected_dim,) for row in result.transitions)
        finally:
            env.close()
    assert len(interfaces) == 3
    assert len(plants) == 3


def test_dataset_action_schema_is_compact(tmp_path):
    result = collect(
        task="three_tank/regulation",
        plant=_plant(1),
        policy="pid",
        episodes=1,
        seed=3,
        output=tmp_path / "h1-only",
        max_steps=2,
    )
    action_schema = result["manifest"]["action_schema"]
    assert action_schema["space"]["shape"] == [4]
    assert [row["name"] for row in action_schema["fields"]] == [
        "pump_P101",
        "valve_V12",
        "valve_V23",
        "heater_H1",
    ]


def test_layout_object_rejects_inconsistent_mapping():
    with pytest.raises(ValueError, match="match their internal slots"):
        ActuatorLayout(("a",), ("b",), (0,))


@pytest.mark.parametrize(
    ("plant_id", "dimension"),
    [
        ("open-cascade-v1", 7),
        ("recirculating-h1-v1", 4),
        ("lab-three-tank-v1", 6),
    ],
)
def test_builtin_action_dimensions_remain_stable(plant_id, dimension):
    env = make_env("three_tank/regulation", plant=plant_id)
    try:
        assert env.action_space.shape == (dimension,)
    finally:
        env.close()

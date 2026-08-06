"""Plant-aware, name-bound PID and MPC profiles for three_tank."""
from __future__ import annotations


_OPEN_PID = {
    "loops": [
        {"actuator": "feed_pump", "output": "level_1", "pid": [8.0, 0.4, 0.0], "bias": 0.25},
        {"actuator": "outlet_valve_1", "output": "level_2", "pid": [6.0, 0.3, 0.0], "bias": 0.22934030538459382},
        {"actuator": "outlet_valve_2", "output": "level_3", "pid": [6.0, 0.3, 0.0], "bias": 0.22934030538459382},
        {"actuator": "heater_H1", "output": "temperature_1", "pid": [0.06, 0.01, 0.0], "bias": 0.37875555555555557},
        {"actuator": "heater_H2", "output": "temperature_2", "pid": [0.06, 0.01, 0.0], "bias": 0.2924},
        {"actuator": "heater_H3", "output": "temperature_3", "pid": [0.06, 0.01, 0.0], "bias": 0.29906666666666665},
    ],
    "holds": [{"actuator": "outlet_valve_3", "value": 0.22934030538459382}],
}

_RECIRCULATING_PID = {
    "loops": [
        {"actuator": "pump_P101", "output": "level_1", "pid": [6.0, 0.2, 0.0], "bias": 0.4507771067833858},
        {"actuator": "valve_V12", "output": "level_2", "pid": [6.0, 0.2, 0.0], "bias": 0.2268046058132572},
        {"actuator": "valve_V23", "output": "level_3", "pid": [6.0, 0.2, 0.0], "bias": 0.2268046058132572},
        {"actuator": "heater_H1", "output": "temperature_1", "pid": [0.08, 0.005, 0.0], "bias": 0.609065572060349},
    ]
}

_LAB_PID = {
    "matrix_terms": [
        {"actuator": "pump_P101", "output": "level_1", "kp": 4.0},
        {"actuator": "valve_V12", "output": "level_1", "kp": -3.0},
        {"actuator": "valve_V12", "output": "level_2", "kp": 3.0},
        {"actuator": "valve_V23", "output": "level_2", "kp": -3.0},
        {"actuator": "valve_V23", "output": "level_3", "kp": 3.0},
        {"actuator": "heater_H1", "output": "temperature_1", "kp": 0.12, "ki": 0.0005},
        {"actuator": "heater_H2", "output": "temperature_2", "kp": 0.12, "ki": 0.0005},
        {"actuator": "heater_H3", "output": "temperature_3", "kp": 0.12, "ki": 0.0005},
    ],
    "bias": "default_action",
}

_MPC = {
    "open-cascade-v1": {"Ts": 0.5, "P": 40, "move_supp": 0.8, "steady_input_weight": 0.0, "q_y": 1.0},
    "recirculating-h1-v1": {"Ts": 0.5, "P": 40, "move_supp": 0.1, "steady_input_weight": 0.0, "q_y": 1.0},
    "lab-three-tank-v1": {"Ts": 1.0, "P": 20, "move_supp": 0.1, "steady_input_weight": 0.0, "q_y": 1.0},
}


def resolve_controller_profile(
    controller_id, *, plant_id, plant, condition_id, objective
):
    del condition_id, objective
    declaration = plant.config.plant
    if declaration["topology"] == "open_cascade":
        family = "open"
    elif "parameters" in declaration and len(declaration["actuators"]) == 4:
        family = "recirculating-h1"
    else:
        family = "lab"
    if controller_id == "pid":
        return {
            "open": _OPEN_PID,
            "recirculating-h1": _RECIRCULATING_PID,
            "lab": _LAB_PID,
        }[family]
    if controller_id == "mpc":
        key = {
            "open": "open-cascade-v1",
            "recirculating-h1": "recirculating-h1-v1",
            "lab": "lab-three-tank-v1",
        }[family]
        return _MPC[key]
    return {}


__all__ = ["resolve_controller_profile"]

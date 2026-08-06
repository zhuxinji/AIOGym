from aiogym.scenarios._shared import build_plugin

from .model import CascadeModel


_PID = {
    "loops": [
        {"u_index": 0, "y_index": 0, "pid": [8.0, 0.4, 0.0], "bias": 0.25},
        {"u_index": 1, "y_index": 1, "pid": [6.0, 0.3, 0.0], "bias": 0.22934030538459382},
        {"u_index": 2, "y_index": 2, "pid": [6.0, 0.3, 0.0], "bias": 0.22934030538459382},
        {"u_index": 4, "y_index": 3, "pid": [0.06, 0.01, 0.0], "bias": 0.37875555555555557},
        {"u_index": 5, "y_index": 4, "pid": [0.06, 0.01, 0.0], "bias": 0.2924},
        {"u_index": 6, "y_index": 5, "pid": [0.06, 0.01, 0.0], "bias": 0.29906666666666665},
    ],
    "holds": [{"u_index": 3, "value": 0.22934030538459382}],
}
_MPC = {"Ts": 0.5, "P": 40, "move_supp": 0.8, "steady_input_weight": 0.0, "q_y": 1.0}


PLUGIN = build_plugin(
    "cascade",
    model_factory=CascadeModel,
    presets=(
        "commissioning",
        "continuous-benchmark",
        "disturbance-rejection",
        "safety-recovery",
        "temperature-step",
    ),
    default_preset="continuous-benchmark",
    economic=True,
    horizon=600,
    control_dt=0.5,
    controller_defaults={
        "pid": {name: _PID for name in (
            "commissioning", "continuous-benchmark", "disturbance-rejection", "safety-recovery", "temperature-step"
        )},
        "mpc": {name: _MPC for name in (
            "commissioning", "continuous-benchmark", "disturbance-rejection", "safety-recovery", "temperature-step"
        )},
    },
    preset_configs={
        "continuous-benchmark": {
            "initial_state": [0.45, 35.0, 0.45, 50.0, 0.45, 65.0],
            "reference": [0.45, 0.45, 0.45, 35.0, 50.0, 65.0],
            "disturbance_schedule": {
                100: {"pump_flow_factor": 0.7},
                400: {"pump_flow_factor": 1.0},
            },
        },
    },
)

__all__ = ["PLUGIN"]

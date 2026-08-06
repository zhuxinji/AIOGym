from aiogym.scenarios._shared import build_plugin

from .model import RecirculatingCascadeModel


_PID = {
    "loops": [
        {"u_index": 0, "y_index": 0, "pid": [6.0, 0.2, 0.0], "bias": 0.4507771067833858},
        {"u_index": 1, "y_index": 1, "pid": [6.0, 0.2, 0.0], "bias": 0.2268046058132572},
        {"u_index": 2, "y_index": 2, "pid": [6.0, 0.2, 0.0], "bias": 0.2268046058132572},
        {"u_index": 3, "y_index": 3, "pid": [0.08, 0.005, 0.0], "bias": 0.609065572060349},
    ]
}
_MPC = {"Ts": 0.5, "P": 40, "move_supp": 0.1, "steady_input_weight": 0.0, "q_y": 1.0}
_PRESETS = (
    "commissioning",
    "hydraulic-commissioning",
    "disturbance-rejection",
    "safety-recovery",
    "temperature-step",
)


PLUGIN = build_plugin(
    "cascade_recirculating",
    model_factory=RecirculatingCascadeModel,
    presets=(
        "commissioning",
        "hydraulic-commissioning",
        "disturbance-rejection",
        "safety-recovery",
        "temperature-step",
    ),
    default_preset="commissioning",
    horizon=600,
    control_dt=0.5,
    controller_defaults={
        "pid": {name: _PID for name in _PRESETS},
        "mpc": {name: _MPC for name in _PRESETS},
    },
    preset_configs={
        "commissioning": {
            "initial_state": [0.18, 24.0, 0.22, 25.5, 0.32, 26.5],
            "reference": [
                0.24,
                0.24,
                0.24,
                30.0,
                28.97128161165881,
                27.654664660905787,
            ],
        },
    },
)

__all__ = ["PLUGIN"]

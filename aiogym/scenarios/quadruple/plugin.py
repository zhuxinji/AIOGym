from aiogym.scenarios._legacy import build_plugin

from .model import QuadrupleModel


_PID = {
    "minimum-phase": {
        "loops": [
            {"u_index": 0, "y_index": 0, "pid": [1.0159437333249033, 0.1, 0.0], "bias": 0.3},
            {"u_index": 1, "y_index": 1, "pid": [2.0, 0.09001698960588032, 0.0], "bias": 0.3},
        ]
    },
    "nonminimum-phase": {
        "loops": [
            {"u_index": 0, "y_index": 1, "pid": [0.02461, 0.0002324, 0.0], "bias": 0.315},
            {"u_index": 1, "y_index": 0, "pid": [0.07719, 0.0003802, 0.0], "bias": 0.315},
        ]
    },
    "zero-boundary-stress": {
        "loops": [
            {"u_index": 0, "y_index": 0, "pid": [1.0, 0.07874058903578529, 0.0], "bias": 0.3},
            {"u_index": 1, "y_index": 1, "pid": [0.9329673736992764, 0.1, 0.0], "bias": 0.3},
        ]
    },
    "disturbance-rejection": {
        "loops": [
            {"u_index": 0, "y_index": 0, "pid": [0.4571, 0.05, 0.0], "bias": 0.3},
            {"u_index": 1, "y_index": 1, "pid": [0.5, 0.05, 0.0], "bias": 0.3},
        ]
    },
}
_MPC = {
    "minimum-phase": {"Ts": 1.0, "P": 1, "move_supp": 0.0, "cv_scale": [1.0, 1.0]},
    "nonminimum-phase": {
        "Ts": 1.0,
        "P": 48,
        "move_supp": 0.0,
        "steady_input_weight": 20000.0,
        "cv_scale": [1.0, 1.0],
    },
    "zero-boundary-stress": {"Ts": 1.0, "P": 1, "move_supp": 0.0, "cv_scale": [1.0, 1.0]},
    "disturbance-rejection": {"Ts": 1.0, "P": 3, "move_supp": 0.0, "cv_scale": [1.0, 1.0]},
}


PLUGIN = build_plugin(
    "quadruple",
    model_factory=QuadrupleModel,
    presets=(
        "minimum-phase",
        "nonminimum-phase",
        "zero-boundary-stress",
        "disturbance-rejection",
    ),
    default_preset="minimum-phase",
    horizon=600,
    control_dt=1.0,
    controller_defaults={"pid": _PID, "mpc": _MPC},
)

__all__ = ["PLUGIN"]

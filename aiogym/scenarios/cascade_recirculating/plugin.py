from aiogym.scenarios._legacy import build_plugin

from .model import RecirculatingCascadeModel


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
        "pid": {"regulation": {}},
        "mpc": {"regulation": {"P": 20, "Ts": 0.5}},
    },
)

__all__ = ["PLUGIN"]

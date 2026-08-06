from aiogym.scenarios._legacy import build_plugin

from .model import CascadeModel


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
        "pid": {"regulation": {"profile": "cascade-control-benchmark"}},
        "mpc": {"regulation": {"profile": "cascade-control-benchmark"}},
    },
)

__all__ = ["PLUGIN"]

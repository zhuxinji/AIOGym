from aiogym.scenarios._legacy import build_plugin

from .model import QuadrupleModel


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
    controller_defaults={
        "pid": {"regulation": {"profile": "quadruple-minimum-phase-benchmark"}},
        "mpc": {"regulation": {"profile": "quadruple-minimum-phase"}},
    },
)

__all__ = ["PLUGIN"]

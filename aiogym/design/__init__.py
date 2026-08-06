"""Parameterised process-design studies for AIO-Gym.

Design studies are intentionally separate from official benchmark Tracks:
changing equipment changes the plant being evaluated and therefore must not
silently change a fixed benchmark identity.
"""
from .model import ThreeTankDesignModel, compile_design_model
from .report import render_design_report, write_design_report_bundle
from .spec import (
    DESIGN_SPEC_SCHEMA_VERSION,
    load_design_spec,
    validate_design_spec,
)
from .study import run_design_study, run_design_sweep
from .wizard import prompt_design_spec

__all__ = [
    "DESIGN_SPEC_SCHEMA_VERSION",
    "ThreeTankDesignModel",
    "compile_design_model",
    "load_design_spec",
    "prompt_design_spec",
    "render_design_report",
    "run_design_study",
    "run_design_sweep",
    "validate_design_spec",
    "write_design_report_bundle",
]

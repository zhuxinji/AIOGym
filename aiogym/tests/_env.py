"""Test-only adapter for concise physical-model fixtures.

Production callers use the strict ``aiogym.make_env`` signature. Physics tests
often need many resolved config fields, so this helper keeps those fixtures
readable while exercising config mode.
"""
from __future__ import annotations

from aiogym.env_factory import make_env as public_make_env


def make_test_env(
    scenario="cascade",
    *,
    case=None,
    reward_spec=None,
    info_level="full",
    **environment,
):
    return public_make_env(
        config={
            "scenario": scenario,
            "case": case,
            "reward_spec": reward_spec,
            "info_level": info_level,
            "environment": environment,
        }
    )


__all__ = ["make_test_env"]

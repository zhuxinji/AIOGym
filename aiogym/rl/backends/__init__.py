"""Private direct-call training backends."""
from __future__ import annotations

from .contracts import BackendResult, validate_backend_result


def run_backend(plan) -> BackendResult:
    """Dispatch a resolved plan without importing optional trainers eagerly."""

    algorithm = plan.config.algorithm_id
    if algorithm in {"ppo", "sac", "td3"}:
        from .sb3 import run_sb3

        result = run_sb3(plan)
    elif algorithm == "rlpd":
        from .rlpd import run_rlpd

        result = run_rlpd(plan)
    elif algorithm == "bc":
        from .bc import run_bc

        result = run_bc(plan)
    else:
        raise ValueError(f"unsupported training algorithm: {algorithm!r}")
    validate_backend_result(plan, result)
    return result


__all__ = ["BackendResult", "run_backend", "validate_backend_result"]

"""Economic and resource-use metrics for evaluation rollouts."""
from __future__ import annotations

from typing import Any, Mapping


def economic_step_metrics(
    info,
    dt: float,
    *,
    env=None,
    state=None,
    action=None,
    disturbance: Mapping[str, Any] | None = None,
):
    """Return reward-independent economic quantities for one transition."""

    if env is None or state is None or action is None:
        return {
            "profit": float(info.get("profit", 0.0)),
            "production": float(info.get("production", 0.0)),
            "energy_kwh": float(info.get("energy_kw", 0.0)) * dt / 3600.0,
            "product_value": float(info.get("product_value", 0.0)),
            "energy_cost": float(info.get("energy_cost", 0.0)),
            "material_cost": float(info.get("material_cost", 0.0)),
            "waste_cost": float(info.get("waste_cost", 0.0)),
        }

    model = env.model
    x = model.state_vector(state)
    u = model.action_vector(action)
    disturbances = dict(disturbance or {})
    cfg = dict(getattr(model, "economic_config", {}))
    energy_kw = float(info.get("energy_kw", 0.0))
    production_rate = 0.0
    production_resolver = getattr(model, "production", None)
    if cfg.get("value") == "production" and callable(production_resolver):
        production_rate = float(production_resolver(x, u, disturbances))
    product_value = (
        float(cfg.get("w_value", 0.0)) * production_rate * dt
    )
    energy_cost = float(cfg.get("w_energy", 0.0)) * energy_kw * dt
    material_cost = _declared_cost(model, "material", x, u, disturbances, dt)
    waste_cost = _declared_cost(model, "waste", x, u, disturbances, dt)
    shortfall = 0.0
    shortfall_resolver = getattr(model, "product_flow_shortfall", None)
    if callable(shortfall_resolver):
        shortfall = max(0.0, float(shortfall_resolver(production_rate)))
    violation = _economic_violation(model, x, info, cfg)
    runaway_cost = 50.0 * dt if bool(info.get("runaway", False)) else 0.0
    profit = (
        product_value
        - energy_cost
        - material_cost
        - waste_cost
        - float(cfg.get("w_viol", 0.0)) * violation * dt
        - float(cfg.get("w_product_shortfall", 0.0)) * shortfall * dt
        - runaway_cost
    )
    return {
        "profit": float(profit),
        "production": float(production_rate * dt),
        "energy_kwh": float(energy_kw * dt / 3600.0),
        "product_value": float(product_value),
        "energy_cost": float(energy_cost),
        "material_cost": float(material_cost),
        "waste_cost": float(waste_cost),
    }


def _declared_cost(model, name, state, action, disturbance, dt):
    resolver = getattr(model, f"{name}_cost_rate", None)
    if not callable(resolver):
        return 0.0
    return max(0.0, float(resolver(state, action, disturbance))) * dt


def _economic_violation(model, state, info, config):
    violation = 0.0
    temps = list(info.get("temps", []))
    for i, (lower, upper) in enumerate(config.get("temp_band", [])):
        value = temps[i] if i < len(temps) else 0.0
        if lower is not None and value < lower:
            violation += (lower - value) / 10.0
        if upper is not None and value > upper:
            violation += (value - upper) / 10.0
    level_scale = max(float(config.get("level_scale", 0.1)), 1e-12)
    controlled = list(model.controlled_output(state))
    for i, (lower, upper) in enumerate(config.get("level_band", [])):
        value = controlled[i] if i < len(controlled) else 0.0
        if lower is not None and value < lower:
            violation += (lower - value) / level_scale
        if upper is not None and value > upper:
            violation += (value - upper) / level_scale
    return float(violation)

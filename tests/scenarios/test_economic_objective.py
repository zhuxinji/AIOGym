from __future__ import annotations

import numpy as np
import pytest

from aiogym.core.env import make_env
from aiogym.core.registry import get_reward
from aiogym.scenarios._shared import economic_reward
from aiogym.workflows import evaluate


class FixedEconomicModel:
    def __init__(self, *, product_flow_m3s, power_kw):
        self.product_flow_m3s = product_flow_m3s
        self.power_kw = power_kw

    def production(self, state, action, disturbances):
        del state, action, disturbances
        return self.product_flow_m3s

    def action_energy_kw(self, action, state, disturbances):
        del action, state, disturbances
        return self.power_kw


def _reward(*, flow, power, dt):
    return economic_reward(
        [0.0],
        [0.0],
        [0.0],
        {
            "model": FixedEconomicModel(
                product_flow_m3s=flow,
                power_kw=power,
            ),
            "disturbances": {},
            "control_dt": dt,
        },
    )


def test_reward_is_dimensional_time_integrated_and_terms_sum():
    reward, terms = _reward(flow=0.002, power=12.0, dt=2.0)
    assert terms["product_value"] == pytest.approx(100000.0 * 0.002 * 2.0)
    assert terms["energy_cost"] == pytest.approx(-(0.7 * 12.0 * 2.0 / 3600.0))
    assert reward == pytest.approx(sum(terms.values()))

    double_dt, _ = _reward(flow=0.002, power=12.0, dt=4.0)
    assert double_dt == pytest.approx(2.0 * reward)
    assert 10 * _reward(flow=0.002, power=12.0, dt=1.0)[0] == pytest.approx(
        5 * _reward(flow=0.002, power=12.0, dt=2.0)[0]
    )


def test_zero_flow_and_zero_power_isolate_the_two_terms():
    energy_only, energy_terms = _reward(flow=0.0, power=9.0, dt=3.0)
    assert energy_terms["product_value"] == 0.0
    assert energy_only == energy_terms["energy_cost"] < 0.0

    product_only, product_terms = _reward(flow=0.001, power=0.0, dt=3.0)
    assert product_terms["energy_cost"] == 0.0
    assert product_only == product_terms["product_value"] > 0.0


def test_economic_reward_has_only_the_public_metric_description():
    reward = get_reward("three_tank", "economic")
    assert reward.primary_metric == "economic_objective"
    assert reward.metric_direction == "maximize"
    assert callable(reward.function)
    assert callable(reward.episode_metric_function)


def test_three_tank_economic_env_reports_integrated_units():
    env = make_env("three_tank", reward="economic")
    try:
        env.reset(seed=0)
        action = np.asarray([0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        terms = info["reward_terms"]
        assert reward == pytest.approx(sum(terms.values()))
        assert info["product_flow_m3s"] > 0.0
        assert terms["product_value"] == pytest.approx(
            100000.0 * info["product_flow_m3s"] * env.control_dt
        )
        assert -terms["energy_cost"] == pytest.approx(
            0.7 * info["energy_kw"] * env.control_dt / 3600.0
        )
    finally:
        env.close()

    env = make_env("three_tank", reward="economic")
    try:
        result = evaluate(env=env, policy="pid", seeds=(0,), max_steps=3)
    finally:
        env.close()
    metrics = result["episodes"][0]["metrics"]
    assert result["schema_version"] == "aiogym.evaluation.v3"
    assert metrics["economic_objective"] == pytest.approx(metrics["return"])
    assert metrics["net_economic_value"] == pytest.approx(metrics["return"])
    assert metrics["product_value"] - metrics["energy_cost"] == pytest.approx(
        metrics["net_economic_value"]
    )
    assert metrics["production_volume_m3"] > 0.0
    assert metrics["energy_kwh"] == pytest.approx(metrics["energy"])


def test_regulation_reward_is_unchanged_by_economic_metrics():
    regulation = get_reward("three_tank", "regulation")
    assert regulation.primary_metric == "tracking_iae"
    assert regulation.metric_direction == "minimize"

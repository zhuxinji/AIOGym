from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aiogym.core import get_task, make_env
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


def _reward(*, flow, power, dt, value=100000.0, price=0.7):
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
            "objective_config": {
                "value_unit": "normalized_value",
                "product_value_per_m3": value,
                "electricity_price_per_kwh": price,
            },
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


def test_economic_task_identity_contains_revision_units_and_coefficients():
    task = get_task("three_tank/economic")
    assert task.revision == 2
    assert task.reward_id == "net-economic-value-v2"
    assert task.metric_suite_id == "economic-core-v2"
    assert task.objective_config == {
        "value_unit": "normalized_value",
        "product_value_per_m3": 100000.0,
        "electricity_price_per_kwh": 0.7,
    }
    with pytest.raises(TypeError):
        task.objective_config["product_value_per_m3"] = 1.0
    changed = replace(
        task,
        objective_config={
            **dict(task.objective_config),
            "electricity_price_per_kwh": 0.8,
        },
    )
    assert changed.task_hash != task.task_hash
    assert task.identity()["objective_config"]["value_unit"] == "normalized_value"


def test_open_cascade_economic_env_reports_integrated_units():
    env = make_env(
        "three_tank/economic",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
    )
    try:
        env.reset(seed=0)
        action = np.asarray([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
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

    result = evaluate(
        "pid",
        task="three_tank/economic",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
        seeds=(0,),
        max_steps=3,
    )
    metrics = result["episodes"][0]["metrics"]
    assert result["schema_version"] == "aiogym.evaluation.v2"
    assert metrics["economic_objective"] == pytest.approx(metrics["return"])
    assert metrics["net_economic_value"] == pytest.approx(metrics["return"])
    assert metrics["product_value"] - metrics["energy_cost"] == pytest.approx(
        metrics["net_economic_value"]
    )
    assert metrics["production_volume_m3"] > 0.0
    assert metrics["energy_kwh"] == pytest.approx(metrics["energy"])


def test_non_product_flow_plant_cannot_create_economic_env():
    with pytest.raises(ValueError, match="requires capability product_flow"):
        make_env("three_tank/economic", plant="recirculating-h1-v1")


class OldEconomicPolicy:
    def __init__(self, contract):
        self.contract = contract

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del observation, context
        return np.zeros(7, dtype=np.float32)

    def metadata(self):
        return {"id": "old-economic", "training_contract": self.contract}


def test_old_economic_training_contract_is_rejected_and_regulation_is_unchanged():
    env = make_env(
        "three_tank/economic",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
    )
    try:
        old_contract = env.identity.as_dict()
    finally:
        env.close()
    old_contract["task_hash"] = "0" * 64
    with pytest.raises(ValueError, match="task_hash"):
        evaluate(
            OldEconomicPolicy(old_contract),
            task="three_tank/economic",
            plant="open-cascade-v1",
            condition="continuous-benchmark",
            max_steps=1,
        )

    regulation = get_task("three_tank/regulation")
    assert dict(regulation.objective_config) == {}
    assert regulation.revision == 1
    assert regulation.reward_id == "normalized-tracking-mse-v1"

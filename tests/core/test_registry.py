from __future__ import annotations

import json

import pytest

from aiogym.core.contracts import Scenario
from aiogym.core.io import write_json
from aiogym.core.registry import (
    get_benchmark,
    get_reward,
    get_scenario,
    list_benchmarks,
    list_parameters,
    list_rewards,
    list_scenarios,
    register_scenario,
    unregister_scenario,
)
from aiogym.core.specs import Benchmark, EpisodeSpec, Reward


class RegistryModel:
    scenario = "registry-toy"
    parameter_units = {}

    def __init__(self, parameters=None):
        if parameters:
            raise ValueError("registry toy has no parameters")
        self.resolved_parameters = {}


def _reward(state, action, next_state, context):
    del state, action, context
    return -abs(float(next_state[0]))


def _metrics(env, episode):
    del env
    return {"return": episode.episode_return}


def _episode(_model):
    return EpisodeSpec(
        initial_state=(0.0,),
        initial_action=(0.0,),
        reference=(0.0,),
        horizon=2,
    )


def _benchmarks():
    return {
        name: Benchmark(
            id=name,
            reward_id="regulation",
            episode_factory=lambda model, rng: _episode(model),
            metric_function=_metrics,
            ranking_metrics=(("return", "maximize"),),
        )
        for name in ("tracking", "disturbance-rejection", "boundary-safety")
    }


def _scenario():
    reward = Reward(
        id="regulation",
        function=_reward,
        episode_metric_function=_metrics,
        primary_metric="return",
        metric_direction="maximize",
    )
    return Scenario(
        id="registry-toy",
        make_model=RegistryModel,
        control_dt=1.0,
        make_default_episode=_episode,
        sample_training_episode=lambda model, rng, reward_id: (
            _episode(model),
            "tracking",
        ),
        sample_training_disturbance=lambda model, rng: {},
        benchmarks=_benchmarks(),
        rewards={"regulation": reward},
        default_reward="regulation",
    )


def test_registry_is_the_single_scenario_reward_and_benchmark_index():
    unregister_scenario("registry-toy")
    scenario = _scenario()
    register_scenario(scenario)
    try:
        assert get_scenario("registry-toy") is scenario
        assert get_reward("registry-toy", "regulation") is scenario.rewards["regulation"]
        assert get_benchmark("registry-toy", "tracking") is scenario.benchmarks[
            "tracking"
        ]
        assert "registry-toy" in list_scenarios()
        assert list_rewards(scenario="registry-toy") == ("regulation",)
        assert list_benchmarks(scenario="registry-toy") == (
            "boundary-safety",
            "disturbance-rejection",
            "tracking",
        )
        assert list_parameters(scenario="registry-toy") == ()
        with pytest.raises(ValueError, match="already registered"):
            register_scenario(scenario)
    finally:
        unregister_scenario("registry-toy")


def test_atomic_json_is_canonical(tmp_path):
    target = write_json(tmp_path / "identity.json", {"b": 2, "a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        write_json(target, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="NaN or Infinity"):
        write_json(tmp_path / "invalid.json", {"value": float("nan")})

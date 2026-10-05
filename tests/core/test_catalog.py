from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aiogym.core.contracts import Scenario
from aiogym.core.io import write_json
from aiogym.core.catalog import (
    get_benchmark,
    get_reward,
    get_scenario,
    list_benchmarks,
    list_parameters,
    list_rewards,
    list_scenarios,
)
from aiogym.core.specs import Benchmark, EpisodeSpec, Reward


class CatalogModel:
    scenario = "catalog-toy"
    parameter_units = {}

    def __init__(self, parameters=None):
        if parameters:
            raise ValueError("catalog toy has no parameters")
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
        id="catalog-toy",
        make_model=CatalogModel,
        control_dt=1.0,
        make_default_episode=_episode,
        sample_training_episode=lambda model, rng, reward_id, boundary: (
            _episode(model),
            "tracking",
        ),
        sample_training_disturbance=lambda model, rng: {},
        benchmarks=_benchmarks(),
        rewards={"regulation": reward},
        default_reward="regulation",
    )


def test_explicit_scenario_resolves_without_changing_builtin_catalog():
    scenario = _scenario()
    assert get_scenario(scenario) is scenario
    assert get_reward(scenario, "regulation") is scenario.rewards["regulation"]
    assert get_benchmark(scenario, "tracking") is scenario.benchmarks[
        "tracking"
    ]
    assert scenario.id not in list_scenarios()
    assert list_rewards(scenario=scenario) == ("regulation",)
    assert list_benchmarks(scenario=scenario) == (
        "boundary-safety",
        "disturbance-rejection",
        "tracking",
    )
    assert list_parameters(scenario=scenario) == ()
    with pytest.raises(KeyError, match="unknown scenario"):
        get_scenario(scenario.id)


def test_atomic_json_is_canonical(tmp_path):
    target = write_json(tmp_path / "identity.json", {"b": 2, "a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        write_json(target, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="NaN or Infinity"):
        write_json(tmp_path / "invalid.json", {"value": float("nan")})


def test_definition_lookup_does_not_construct_models_or_episodes():
    def unexpected_factory(*args):
        raise AssertionError("lookup must not perform numerical initialization")

    original = _scenario()
    scenario = replace(
        original,
        make_model=unexpected_factory,
        make_default_episode=unexpected_factory,
        benchmarks={
            name: replace(benchmark, episode_factory=unexpected_factory)
            for name, benchmark in original.benchmarks.items()
        },
    )
    assert get_scenario(scenario) is scenario
    assert list_rewards(scenario=scenario) == ("regulation",)
    assert len(list_benchmarks(scenario=scenario)) == 3

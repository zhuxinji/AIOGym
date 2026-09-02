from __future__ import annotations

from types import MappingProxyType

import pytest

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Benchmark, EpisodeSpec, Reward


def _reward(*_):
    return 0.0


def _metrics(env, episode):
    del env
    return {"return": episode.episode_return}


class ImmutableModel:
    scenario = "immutable"

    def __init__(self, parameters=None):
        if parameters:
            raise ValueError("immutable toy has no parameters")


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


def test_episode_inputs_and_nested_schedules_are_defensively_immutable():
    disturbances = {"ambient": 18.0}
    schedule = {2: {"ambient": 17.0}}
    episode = EpisodeSpec(
        initial_state=(0.1, 20.0),
        initial_action=(0.0,),
        reference=(0.2, 25.0),
        horizon=10,
        disturbances=disturbances,
        disturbance_schedule=schedule,
    )
    disturbances["ambient"] = 99.0
    schedule[2]["ambient"] = 99.0
    assert episode.disturbances["ambient"] == 18.0
    assert episode.disturbance_schedule[2]["ambient"] == 17.0
    with pytest.raises(TypeError):
        episode.disturbances["ambient"] = 10.0


def test_scenario_benchmarks_and_rewards_are_read_only_mappings():
    reward = Reward(id="regulation", function=_reward)
    scenario = Scenario(
        id="immutable",
        make_model=ImmutableModel,
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
    assert isinstance(scenario.benchmarks, MappingProxyType)
    assert isinstance(scenario.rewards, MappingProxyType)
    with pytest.raises(TypeError):
        scenario.benchmarks["other"] = scenario.benchmarks["tracking"]
    with pytest.raises(TypeError):
        scenario.rewards["other"] = reward


def test_scenario_rejects_a_benchmark_with_an_unknown_reward():
    benchmarks = _benchmarks()
    benchmark = benchmarks["tracking"]
    benchmarks["tracking"] = Benchmark(
        id=benchmark.id,
        reward_id="missing",
        episode_factory=benchmark.episode_factory,
        metric_function=benchmark.metric_function,
        ranking_metrics=benchmark.ranking_metrics,
    )
    with pytest.raises(ValueError, match="benchmark reward_id"):
        Scenario(
            id="immutable",
            make_model=ImmutableModel,
            control_dt=1.0,
            make_default_episode=_episode,
            sample_training_episode=lambda model, rng, reward_id, boundary: (
                _episode(model),
                "tracking",
            ),
            sample_training_disturbance=lambda model, rng: {},
            benchmarks=benchmarks,
            rewards={"regulation": Reward(id="regulation", function=_reward)},
            default_reward="regulation",
        )

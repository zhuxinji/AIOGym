from __future__ import annotations

import pytest

from aiogym.core.specs import Benchmark, EpisodeSpec, Reward


def _reward(state, action, next_state, context):
    del state, action, context
    return -float(next_state[0])


def _metrics(env, episode):
    del env
    return {"return": episode.episode_return}


def test_episode_spec_contains_only_resolved_episode_data():
    episode = EpisodeSpec(
        initial_state=(0.2, 20.0),
        initial_action=(0.5,),
        reference=(0.3,),
        horizon=10,
        disturbances={"gain": 1.0},
        reference_schedule={5: (0.4,)},
        disturbance_schedule={7: {"gain": 0.8}},
    )
    assert set(episode.as_dict()) == {
        "initial_state",
        "initial_action",
        "reference",
        "horizon",
        "disturbances",
        "reference_schedule",
        "disturbance_schedule",
    }
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        EpisodeSpec(**episode.as_dict(), controller="sac")


def test_benchmark_declares_fixed_episode_metrics_and_ranking():
    benchmark = Benchmark(
        id="tracking",
        reward_id="regulation",
        episode_factory=lambda model, rng: EpisodeSpec(
            initial_state=(0.0,),
            initial_action=(0.0,),
            reference=(0.0,),
            horizon=2,
        ),
        metric_function=_metrics,
        ranking_metrics=(("return", "maximize"),),
    )
    assert benchmark.ranking_metrics == (("return", "maximize"),)
    assert benchmark.reward_id == "regulation"

    with pytest.raises(ValueError, match="reward_id"):
        Benchmark(
            id="tracking",
            reward_id="",
            episode_factory=benchmark.episode_factory,
            metric_function=_metrics,
            ranking_metrics=(("return", "maximize"),),
        )


def test_benchmark_case_seed_deterministically_resolves_episode():
    benchmark = Benchmark(
        id="tracking",
        reward_id="regulation",
        episode_factory=lambda model, rng: EpisodeSpec(
            initial_state=(float(rng.uniform()),),
            initial_action=(0.0,),
            reference=(0.0,),
            horizon=2,
        ),
        metric_function=_metrics,
        ranking_metrics=(("return", "maximize"),),
    )
    assert benchmark.make_episode(None, 7) == benchmark.make_episode(None, 7)
    assert benchmark.make_episode(None, 7) != benchmark.make_episode(None, 8)
    with pytest.raises(TypeError, match="case_seed"):
        benchmark.make_episode(None, True)
    with pytest.raises(ValueError, match="case_seed"):
        benchmark.make_episode(None, -1)


def test_reward_is_runtime_function_metric_metadata_and_safety_penalty():
    reward = Reward(
        id="regulation",
        function=_reward,
        episode_metric_function=_metrics,
        primary_metric="return",
        metric_direction="maximize",
        safety_violation_penalty=25.0,
    )
    assert reward.function is _reward
    assert reward.episode_metric_function is _metrics
    assert reward.primary_metric == "return"
    assert reward.metric_direction == "maximize"
    assert reward.safety_violation_penalty == 25.0


def test_reward_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="local name"):
        Reward(id="not/local", function=_reward)
    with pytest.raises(TypeError, match="callable"):
        Reward(id="reward", function=1)
    with pytest.raises(ValueError, match="episode_metric_function"):
        Reward(id="reward", function=_reward, primary_metric="return")
    with pytest.raises(ValueError, match="metric_direction"):
        Reward(id="reward", function=_reward, metric_direction="sideways")
    with pytest.raises(ValueError, match="non-negative"):
        Reward(id="reward", function=_reward, safety_violation_penalty=-1)

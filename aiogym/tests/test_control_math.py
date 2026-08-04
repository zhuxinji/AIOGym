from __future__ import annotations

import numpy as np
import pytest

from aiogym._internal.control_math import (
    normalized_action,
    normalized_tracking_errors,
)
from aiogym.evaluation.scorecard import _action_integrals
from aiogym.rewards.terms import tracking_cost_terms


class _Model:
    def action_vector(self, action):
        return list(action)

    def action_schema(self):
        return [
            {"name": "cooling", "bounds": (-10.0, 10.0)},
            {"name": "pressure", "bounds": (100.0, 200.0)},
        ]

    def controlled_output_scales(self):
        return [10.0, 20.0]


def test_non_unit_action_bounds_have_one_normalization_contract():
    model = _Model()
    assert normalized_action(model, [0.0, 150.0]) == pytest.approx([0.5, 0.5])
    terms = tracking_cost_terms(
        model,
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 150.0],
        [-10.0, 100.0],
        [1.0, 1.0],
        1.0,
    )
    slew, effort = _action_integrals(
        model, [0.0, 150.0], [-10.0, 100.0], 1.0
    )
    assert terms[2] == pytest.approx(0.5)
    assert slew == pytest.approx(0.25)
    assert effort == pytest.approx(0.25)


def test_action_schema_dimension_mismatch_fails_explicitly():
    model = _Model()
    model.action_schema = lambda: [{"bounds": (0.0, 1.0)}]
    with pytest.raises(ValueError, match="dimensions differ"):
        normalized_action(model, [0.0, 150.0])


def test_tracking_normalization_preserves_signed_scale_semantics():
    assert normalized_tracking_errors(_Model(), [15.0, 0.0], [5.0, 10.0]) == [
        1.0,
        -0.5,
    ]


def test_integral_observation_uses_per_channel_limits():
    from aiogym.tests._env import make_test_env

    env = make_test_env(
        "cascade",
        integral_obs=True,
        randomize=False,
        randomize_setpoints=False,
        auto_events=False,
    )
    try:
        observation, _ = env.reset(seed=7)
        runtime = env.unwrapped
        assert runtime._integral_limits == (
            8.0,
            8.0,
            8.0,
            300.0,
            300.0,
            300.0,
        )
        runtime._iy = list(runtime._integral_limits)
        observation = runtime._obs()
        assert observation[-6:] == pytest.approx(np.ones(6))
        assert env.observation_space.contains(observation)
        runtime._iy = [7.99, 7.99, 7.99, 299.99, 299.99, 299.99]
        runtime._accumulate_integral()
        assert np.all(
            np.abs(runtime._iy) <= np.asarray(runtime._integral_limits)
        )
    finally:
        env.close()

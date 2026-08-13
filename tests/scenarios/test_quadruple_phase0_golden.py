from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aiogym
from aiogym.core.specs import EpisodeSpec


GOLDEN = json.loads(
    (Path(__file__).with_name("golden") / "quadruple-phase0-v1.json").read_text(
        encoding="utf-8"
    )
)


def test_quadruple_phase0_fixture_is_reproducible():
    env = aiogym.make_env("quadruple", reward="regulation")
    try:
        source = GOLDEN["condition"]
        episode = EpisodeSpec(
            initial_state=tuple(source["initial_state"]),
            reference=tuple(source["reference"]),
            horizon=source["horizon"],
        )
        observation, _ = env.reset(
            seed=GOLDEN["seed"], options={"episode": episode}
        )
        assert observation == pytest.approx(GOLDEN["initial_observation"], abs=1e-8)

        action = np.asarray(GOLDEN["fixed_action"], dtype=np.float32)
        for expected in GOLDEN["steps"]:
            observation, reward, terminated, truncated, info = env.step(action)
            assert info["true_state"] == pytest.approx(expected["state"], abs=1e-12)
            assert observation == pytest.approx(expected["observation"], abs=1e-8)
            assert reward == pytest.approx(expected["reward"], abs=1e-20)
            assert info["constraint_costs"] == expected["constraints"]
            assert terminated is expected["terminated"]
            assert truncated is expected["truncated"]
    finally:
        env.close()

from __future__ import annotations

import json
from pathlib import Path

import aiogym
import pytest


GOLDEN = json.loads(
    (Path(__file__).with_name("golden") / "cascade-default-v8.json").read_text(
        encoding="utf-8"
    )
)


def test_cascade_default_episode_matches_the_versioned_fixture():
    env = aiogym.make_env("cascade")
    try:
        assert GOLDEN["schema_version"] == "aiogym.cascade_default_golden.v8"
        actual_episode = env.unwrapped.default_episode.as_dict()
        expected_episode = GOLDEN["episode"]
        assert actual_episode.keys() == expected_episode.keys()
        approximate_fields = {"initial_state", "initial_action", "reference"}
        for field in approximate_fields:
            assert actual_episode[field] == pytest.approx(
                expected_episode[field], rel=1e-12, abs=1e-12
            )
        assert {
            key: value
            for key, value in actual_episode.items()
            if key not in approximate_fields
        } == {
            key: value
            for key, value in expected_episode.items()
            if key not in approximate_fields
        }
        observation, _ = env.reset(seed=0)
        assert observation == pytest.approx(GOLDEN["interface"]["observation"])
        assert [
            row["name"] for row in env.unwrapped.model.observation_schema()
        ] == GOLDEN["interface"]["observation_names"]
        assert [
            row["name"] for row in env.unwrapped.model.action_schema()
        ] == GOLDEN["interface"]["action_names"]
    finally:
        env.close()

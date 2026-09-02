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
        assert env.unwrapped.default_episode.as_dict() == GOLDEN["episode"]
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

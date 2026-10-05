from __future__ import annotations

import copy

import gymnasium as gym
import pytest

from aiogym import make_env
from aiogym.workflows._metadata import (
    environment_metadata,
    validate_environment_compatibility,
)


def test_equal_shapes_do_not_imply_equal_channel_semantics():
    env = make_env("heater")
    try:
        metadata = environment_metadata(env)
        validate_environment_compatibility(metadata, env)
        reordered = copy.deepcopy(metadata)
        reordered["policy_interface"]["observation"].reverse()
        changed_unit = copy.deepcopy(metadata)
        changed_unit["policy_interface"]["action"][0]["unit"] = "different-unit"
        changed_version = copy.deepcopy(metadata)
        changed_version["policy_interface"]["version"] += 1
        for incompatible in (reordered, changed_unit, changed_version):
            with pytest.raises(ValueError, match="policy_interface is incompatible"):
                validate_environment_compatibility(incompatible, env)
    finally:
        env.close()


def test_wrapper_must_declare_its_actual_interface():
    env = make_env("heater")
    try:
        with pytest.raises(TypeError, match="explicitly implement policy_interface"):
            environment_metadata(gym.Wrapper(env))

        class WrongBounds(gym.Wrapper):
            def policy_interface(self):
                interface = self.env.policy_interface()
                interface["action"][0]["high"] = 0.5
                return interface

        with pytest.raises(ValueError, match="action high does not match space"):
            environment_metadata(WrongBounds(env))
    finally:
        env.close()

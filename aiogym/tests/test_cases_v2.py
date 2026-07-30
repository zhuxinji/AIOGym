from __future__ import annotations

import pytest

import aiogym
from aiogym.tests._env import make_test_env
from aiogym.models.cases import (
    CASE_PROFILE_SCHEMA_VERSION,
    CaseSpec,
    apply_case_overrides,
    case_profile_hash,
    list_cases,
    load_case,
)


def test_bundled_cases_are_v2_and_do_not_own_objectives():
    assert aiogym.list_cases() == list_cases()
    assert aiogym.list_cases()
    for case_id in aiogym.list_cases():
        profile = aiogym.load_case(case_id)
        assert profile["schema_version"] == CASE_PROFILE_SCHEMA_VERSION
        assert not {
            "default_objective",
            "supported_objectives",
            "objectives",
            "controllers",
        }.intersection(profile)


def test_case_hash_is_key_order_independent_and_case_spec_is_defensive():
    profile = load_case("quadruple/minimum-phase")
    reordered = dict(reversed(tuple(profile.items())))
    assert case_profile_hash(profile) == case_profile_hash(reordered)

    spec = CaseSpec(profile)
    mutable_view = spec.as_dict()
    mutable_view["environment"]["episode_steps"] = 1
    assert spec.as_dict()["environment"]["episode_steps"] != 1
    assert spec.profile_hash == case_profile_hash(profile)


def test_case_overrides_are_whitelisted_validated_and_hashed():
    base = load_case("quadruple/minimum-phase")
    resolved = apply_case_overrides(
        base,
        {
            "environment": {"episode_steps": 17},
            "disturbances": [
                {"at_step": 3, "name": "pump_scale", "value": [0.9, 0.9]}
            ],
        },
    )

    assert resolved["environment"]["episode_steps"] == 17
    assert base["environment"]["episode_steps"] != 17
    assert case_profile_hash(resolved) != case_profile_hash(base)
    with pytest.raises(ValueError, match="may not change fields: name"):
        apply_case_overrides(base, {"name": "not-the-same-case"})
    with pytest.raises(ValueError, match="may not change fields: objective"):
        apply_case_overrides(base, {"objective": "tracking"})


def test_case_environment_path_rejects_retired_task_alias():
    env = make_test_env(
        "quadruple",
        case="minimum-phase",
        episode_steps=1,
    )
    assert env.case_profile["schema_version"] == CASE_PROFILE_SCHEMA_VERSION
    env.close()

    with pytest.raises(TypeError, match="unexpected keyword argument 'task'"):
        aiogym.make_env(
            "quadruple",
            task="minimum-phase",
        )

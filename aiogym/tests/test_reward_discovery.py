from __future__ import annotations

import aiogym

from aiogym.rewards.registry import (
    get_reward_spec,
    resolve_reward_spec,
    reward_display,
)


def test_reward_selectors_return_canonical_registry_instances():
    assert resolve_reward_spec("regulation") is get_reward_spec("regulation-v1")
    assert resolve_reward_spec("economic") is get_reward_spec("economic-v1")


def test_reward_display_metadata_does_not_change_canonical_hash():
    spec = get_reward_spec("regulation-v1")
    before = spec.spec_hash
    display = reward_display("regulation")
    display["term_descriptions"]["tracking_error"] = "changed display"
    assert spec.spec_hash == before
    assert "summary" not in spec.as_dict()


def test_reward_selector_and_canonical_id_have_same_environment_identity():
    short = aiogym.make_env(
        "cascade",
        case="continuous-benchmark",
        reward_spec="economic",
    )
    exact = aiogym.make_env(
        "cascade",
        case="continuous-benchmark",
        reward_spec="economic-v1",
    )
    try:
        assert short.reward_spec is exact.reward_spec
        assert short.env_spec.spec_hash == exact.env_spec.spec_hash
    finally:
        short.close()
        exact.close()

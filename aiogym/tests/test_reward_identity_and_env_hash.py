from __future__ import annotations

from copy import deepcopy

import pytest

from aiogym._environment.spec import (
    ENV_SPEC_HASH_SCHEMA_VERSION,
    resolve_env_spec,
)
from aiogym.benchmarks import load_track
from aiogym.evaluation.provenance import track_provenance
from aiogym.rewards import get_reward_spec
from aiogym.rewards.registry import resolve_reward_spec
from aiogym.rewards.specs import RewardSpec
from aiogym.tests._env import make_test_env


def _equivalent(spec: RewardSpec, **changes) -> RewardSpec:
    payload = deepcopy(spec.as_dict())
    payload.update(changes)
    return RewardSpec(**payload)


def test_same_official_id_with_different_weights_is_rejected():
    canonical = get_reward_spec("regulation-v1")
    weights = dict(canonical.term_weights)
    weights["tracking_error"] = 999.0
    counterfeit = _equivalent(canonical, term_weights=weights)

    assert counterfeit.spec_hash != canonical.spec_hash
    with pytest.raises(ValueError, match="does not match"):
        resolve_reward_spec(counterfeit)


def test_equivalent_reward_object_resolves_to_registry_instance():
    canonical = get_reward_spec("regulation-v1")
    equivalent = _equivalent(canonical)

    assert resolve_reward_spec(equivalent) is canonical


def test_string_and_equivalent_object_produce_same_env_hash():
    canonical = get_reward_spec("regulation-v1")
    by_id = resolve_env_spec("quadruple", reward_spec=canonical.id)
    by_value = resolve_env_spec(
        "quadruple",
        reward_spec=_equivalent(canonical),
    )

    assert by_id.spec_hash == by_value.spec_hash


def test_reward_mapping_order_does_not_change_hash():
    canonical = get_reward_spec("economic-v1")
    reversed_terms = dict(reversed(tuple(canonical.term_weights.items())))
    reversed_costs = dict(reversed(tuple(canonical.cost_weights.items())))
    reordered = _equivalent(
        canonical,
        term_weights=reversed_terms,
        cost_weights=reversed_costs,
    )

    assert reordered.spec_hash == canonical.spec_hash


def test_reward_content_is_present_in_env_hash_payload_contract():
    spec = resolve_env_spec("quadruple", reward_spec="regulation-v1")
    payload = spec.hash_payload()

    assert payload["hash_schema"] == ENV_SPEC_HASH_SCHEMA_VERSION
    assert payload["reward_spec"] == {
        "id": spec.reward_spec.id,
        "spec_hash": spec.reward_spec.spec_hash,
        "content": spec.reward_spec.as_dict(),
    }


def test_environment_and_training_provenance_include_reward_spec_hash():
    env = make_test_env(
        "quadruple",
        reward_spec="regulation-v1",
        episode_steps=1,
        auto_events=False,
    )
    try:
        _, info = env.reset(seed=1)
        assert info["reward_spec_hash"] == env.reward_spec.spec_hash
        assert info["env_spec_hash"] == env.env_spec.spec_hash
        assert info["env_spec_hash_schema"] == ENV_SPEC_HASH_SCHEMA_VERSION
    finally:
        env.close()

    track = load_track("quadruple-regulation-generalist-v1")
    provenance = track_provenance(track, include_test_split=False)
    assert provenance["reward_spec_hash"] == get_reward_spec(
        track.reward_spec_id
    ).spec_hash

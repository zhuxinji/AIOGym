from __future__ import annotations

import numpy as np
import pytest

import aiogym


def test_load_policy_rejects_non_aiogym_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a checkpoint")
    env = aiogym.make_env("quadruple")
    try:
        with pytest.raises(ValueError, match="valid AIO-Gym"):
            aiogym.load_policy(checkpoint, env=env)
    finally:
        env.close()


def test_function_policy_resets_and_records_checked_declaration(tmp_path):
    with aiogym.make_env("heater", randomize=True) as env:
        resets, contexts = [], []
        state = {"step": 0}
        metadata = {
            "id": "external_stateful",
            "environment": aiogym.environment_metadata(env),
            "information_access": ["observation", "context"],
        }

        def reset(*, seed):
            resets.append(seed)
            state["step"] = 0

        def act(observation, context):
            contexts.append(context)
            state["step"] += 1
            return np.full(env.action_space.shape, 0.4 + 0.01 * state["step"])

        policy = aiogym.FunctionPolicy(act, reset=reset, metadata=metadata)
        metadata["environment"]["scenario"] = "modified_after_construction"
        assert isinstance(policy, aiogym.Policy)
        report = aiogym.evaluate(
            env=env, policies={"external": policy, "hold": "hold"},
            seeds=[7, 8], max_steps=2, output=tmp_path / "comparison",
        )
        result = report["evaluations"]["external"]
        assert resets == [7, 8]
        assert [row["step_index"] for row in contexts] == [0, 1, 0, 1]
        assert set(contexts[0]) == {"step_index", "physical_time", "reference"}
        assert result["policy"]["declared_environment_check"] == "matched"
        assert result["policy"]["information_access"] == ["observation", "context"]
        assert result["episodes"][0]["trajectory"]["commanded_action"][0] == pytest.approx([0.41, 0.41])
        assert "step-limited: max_steps=2" in (tmp_path / "comparison/comparison.svg").read_text()
        collected = aiogym.collect(
            env=env, policy=policy, episodes=1, max_steps=1, output=tmp_path / "dataset",
        )
        assert collected["metadata"]["policy"]["declared_environment_check"] == "matched"


def test_external_declaration_mismatch_fails_before_rollout_or_dataset(tmp_path, monkeypatch):
    with aiogym.make_env("heater") as env:
        expected = aiogym.environment_metadata(env)
        expected["policy_interface"]["action"].reverse()
        policy = aiogym.FunctionPolicy(
            lambda observation, context: [0.4, 0.7], metadata={"environment": expected},
        )

        def unexpected_reset(**kwargs):
            raise AssertionError("incompatible policy reached reset")

        monkeypatch.setattr(env, "reset", unexpected_reset)
        with pytest.raises(ValueError, match="policy policy_interface is incompatible"):
            aiogym.evaluate(env=env, policies={"policy": policy}, seeds=[0], max_steps=1)
        with pytest.raises(ValueError, match="policy policy_interface is incompatible"):
            aiogym.collect(env=env, policy=policy, output=tmp_path / "dataset")
        assert not (tmp_path / "dataset").exists()


def test_function_policy_without_declaration_does_not_claim_compatibility_or_clip():
    with aiogym.make_env("heater") as env:
        policy = aiogym.FunctionPolicy(lambda observation, context: [0.4, 0.7])
        result = aiogym.evaluate(env=env, policies={"policy": policy}, seeds=[0], max_steps=1)["evaluations"]["policy"]
        assert result["policy"]["declared_environment_check"] == "not_provided"
        invalid = aiogym.FunctionPolicy(lambda observation, context: [2.0, 0.7])
        with pytest.raises(ValueError, match="policy output must belong directly"):
            aiogym.evaluate(env=env, policies={"policy": invalid}, seeds=[0], max_steps=1)


def test_observation_function_works_in_all_public_policy_workflows(tmp_path):
    observations = []

    def act(observation):
        observations.append(observation.copy())
        return [0.4, 0.75]

    with aiogym.make_env("heater", benchmark="tracking") as env:
        evaluation = aiogym.evaluate(env=env, policies={"policy": act}, seeds=[0], max_steps=2)["evaluations"]["policy"]
        comparison = aiogym.evaluate(
            env=env, policies={"mine": act, "hold": "hold"}, seeds=[0],
            max_steps=2, output=tmp_path / "comparison",
        )
        collected = aiogym.collect(
            env=env, policy=act, episodes=1, seed=0,
            max_steps=2, output=tmp_path / "dataset",
        )
    assert len(observations) == 6
    assert all(row.shape == (5,) for row in observations)
    assert evaluation["policy"]["declared_environment_check"] == "not_provided"
    assert comparison["evaluations"]["mine"]["aggregate"] == evaluation["aggregate"]
    episode = aiogym.DatasetReader(collected["path"]).load_episode(0)
    np.testing.assert_allclose(episode.array("action"), [[0.4, 0.75]] * 2)

from __future__ import annotations

import copy

import numpy as np
import pytest

from aiogym.controllers import make_controller
from aiogym.core import make_env
from aiogym.workflows import collect, evaluate, load_plant


class ContractPolicy:
    def __init__(self, action_dim, contract=None):
        self.action_dim = action_dim
        self.contract = contract

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del observation, context
        return np.zeros(self.action_dim, dtype=np.float32)

    def metadata(self):
        metadata = {"id": "contract-policy"}
        if self.contract is not None:
            metadata["training_contract"] = self.contract
        return metadata


def _changed_condition():
    base = load_plant("recirculating-h1-v1").conditions["commissioning"]
    changed = base.as_dict(include_hash=False)
    changed["id"] = "contract-transfer"
    changed["reference"] = [*base.reference[:3], 31.0, *base.reference[4:]]
    return changed


def test_bound_policy_requires_every_explicit_selector_to_match(tmp_path):
    env = make_env("three_tank/regulation", plant="recirculating-h1-v1")
    try:
        policy = make_controller("pid", env=env)
        same = evaluate(
            policy,
            task="three_tank/regulation",
            plant="recirculating-h1-v1",
            condition="commissioning",
            seeds=(0,),
            max_steps=1,
        )
        assert same["contract_status"] == "bound"
        collected = collect(
            task="three_tank/regulation",
            plant="recirculating-h1-v1",
            condition="commissioning",
            policy=policy,
            episodes=1,
            output=tmp_path / "same",
            max_steps=1,
        )
        assert collected["contract_status"] == "bound"

        with pytest.raises(ValueError, match="bound policy condition_hash"):
            evaluate(
                policy,
                task="three_tank/regulation",
                condition=_changed_condition(),
                max_steps=1,
            )
        with pytest.raises(ValueError, match="bound policy condition_hash"):
            collect(
                task="three_tank/regulation",
                condition=_changed_condition(),
                policy=policy,
                episodes=1,
                output=tmp_path / "different-condition",
                max_steps=1,
            )
        with pytest.raises(ValueError, match="bound policy plant_hash"):
            evaluate(
                policy,
                task="three_tank/regulation",
                plant="lab-three-tank-v1",
                max_steps=1,
            )
    finally:
        env.close()


def test_checkpoint_contract_is_strict_and_transfer_is_opt_in(tmp_path):
    source = make_env("three_tank/regulation", plant="recirculating-h1-v1")
    try:
        contract = source.identity.as_dict()
    finally:
        source.close()
    policy = ContractPolicy(4, contract)

    condition = _changed_condition()
    with pytest.raises(ValueError, match="allow_condition_transfer"):
        evaluate(
            policy,
            task="three_tank/regulation",
            plant="recirculating-h1-v1",
            condition=condition,
            max_steps=1,
        )
    transferred = evaluate(
        policy,
        task="three_tank/regulation",
        plant="recirculating-h1-v1",
        condition=condition,
        max_steps=1,
        allow_condition_transfer=True,
    )
    assert transferred["contract_status"] == "transfer"
    assert transferred["transfer_flags"] == {
        "is_transfer": True,
        "plant_changed": False,
        "condition_changed": True,
    }

    plant = load_plant("recirculating-h1-v1").as_dict(include_hash=False)
    plant["id"] = "same-interface-new-physics"
    plant["plant"]["parameters"]["pump_flow_max"] *= 1.01
    with pytest.raises(ValueError, match="allow_plant_transfer"):
        evaluate(
            policy,
            task="three_tank/regulation",
            plant=plant,
            condition="commissioning",
            max_steps=1,
        )
    collected = collect(
        task="three_tank/regulation",
        plant=plant,
        condition="commissioning",
        policy=policy,
        episodes=1,
        output=tmp_path / "plant-transfer",
        max_steps=1,
        allow_plant_transfer=True,
    )
    assert collected["contract_status"] == "transfer"
    assert collected["manifest"]["policy_training_contract"] == contract
    assert collected["manifest"]["target_environment_contract"] == collected[
        "target_environment_contract"
    ]
    assert collected["manifest"]["transfer_flags"]["plant_changed"]


def test_task_and_interface_never_transfer_and_custom_policy_is_unverified():
    source = make_env("three_tank/regulation", plant="recirculating-h1-v1")
    try:
        contract = source.identity.as_dict()
    finally:
        source.close()

    wrong_task = copy.deepcopy(contract)
    wrong_task["task_hash"] = "0" * 64
    with pytest.raises(ValueError, match="task_hash"):
        evaluate(
            ContractPolicy(4, wrong_task),
            task="three_tank/regulation",
            plant="recirculating-h1-v1",
            max_steps=1,
        )
    with pytest.raises(ValueError, match="interface_hash"):
        evaluate(
            ContractPolicy(4, contract),
            task="three_tank/regulation",
            plant="lab-three-tank-v1",
            max_steps=1,
            allow_plant_transfer=True,
            allow_condition_transfer=True,
        )

    result = evaluate(
        ContractPolicy(4),
        task="three_tank/regulation",
        plant="recirculating-h1-v1",
        seeds=(0,),
        max_steps=1,
    )
    assert result["contract_status"] == "unverified"
    assert result["policy_training_contract"] is None
    assert result["target_environment_contract"]["env_hash"] == result["env_hash"]

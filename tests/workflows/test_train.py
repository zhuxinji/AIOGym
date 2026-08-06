from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from aiogym.workflows import TrainConfig, evaluate, load_checkpoint, train


SMALL_POLICY = {"policy_kwargs": {"net_arch": [8, 8]}}


@pytest.mark.parametrize("algorithm", ("sac", "ppo", "td3", "ddpg"))
def test_supported_algorithms_train_save_reload_and_evaluate(tmp_path, algorithm):
    output = tmp_path / algorithm
    result = train(
        task="quadruple/regulation",
        preset="minimum-phase",
        algorithm=algorithm,
        steps=2,
        seed=1,
        eval_seeds=(10,),
        eval_max_steps=2,
        algorithm_kwargs=SMALL_POLICY,
        output=output,
    )
    checkpoint = Path(result["checkpoint"]["path"])
    contract = Path(result["checkpoint_contract"]["path"])
    assert checkpoint.is_file()
    assert contract.is_file()
    assert len(result["checkpoint"]["sha256"]) == 64
    assert result["evaluation"]["seeds"] == [10]
    assert np.isfinite(result["evaluation"]["aggregate"]["return"]["mean"])
    loaded = load_checkpoint(checkpoint, algorithm=algorithm, device="cpu")
    replay = evaluate(
        loaded,
        task="quadruple/regulation",
        preset="minimum-phase",
        seeds=(11,),
        max_steps=2,
    )
    assert replay["policy"]["algorithm"] == algorithm
    assert replay["transfer_flags"]["is_transfer"] is False
    assert np.isfinite(replay["aggregate"]["return"]["mean"])
    assert "use_sde" not in SMALL_POLICY["policy_kwargs"]
    assert (output / "manifest.json").is_file()
    with pytest.raises(FileExistsError):
        train(
            task="quadruple/regulation",
            preset="minimum-phase",
            algorithm=algorithm,
            steps=1,
            output=output,
        )


@pytest.mark.parametrize(
    ("task", "preset"),
    (
        ("cascade/regulation", "continuous-benchmark"),
        ("three_tank/regulation", "commissioning"),
    ),
)
def test_sac_smoke_runs_on_remaining_stable_scenarios(tmp_path, task, preset):
    result = train(
        task=task,
        preset=preset,
        algorithm="sac",
        steps=2,
        seed=2,
        eval_seeds=(12,),
        eval_max_steps=2,
        algorithm_kwargs=SMALL_POLICY,
        output=tmp_path / task.split("/")[0],
    )
    assert result["algorithm"] == "sac"
    assert result["evaluation"]["episodes"][0]["steps"] == 2
    assert np.isfinite(result["metrics"]["return"]["mean"])


def test_train_config_is_small_and_seed_validation_is_explicit(tmp_path):
    config = TrainConfig(
        task="quadruple/regulation",
        algorithm="sac",
        steps=10,
        output=tmp_path / "run",
        eval_seeds=(3, 4),
    )
    assert config.algorithm == "sac"
    assert set(config.__dataclass_fields__) == {
        "task",
        "algorithm",
        "steps",
        "output",
        "plant",
        "condition",
        "seed",
        "eval_seeds",
        "algorithm_kwargs",
        "eval_max_steps",
    }
    with pytest.raises(ValueError, match="duplicates"):
        TrainConfig(
            task="quadruple/regulation",
            algorithm="sac",
            steps=1,
            output=tmp_path / "bad",
            eval_seeds=(1, 1),
        )
    with pytest.raises(ValueError, match="algorithm"):
        TrainConfig(
            task="quadruple/regulation",
            algorithm="rlpd",
            steps=1,
            output=tmp_path / "bad-algorithm",
        )


def test_importing_training_workflow_does_not_eagerly_load_sb3():
    code = (
        "import sys; import aiogym.workflows.train; "
        "assert 'stable_baselines3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

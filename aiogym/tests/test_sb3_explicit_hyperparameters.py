from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from gymnasium import spaces

from aiogym.rl.config import RLTrainingConfig, resolve_training_defaults
from aiogym.rl.backends.sb3 import (
    _effective_algorithm_kwargs,
    _sb3_training_diagnostics,
)
from aiogym.rl.backends.sb3_algorithms import get_algorithm_adapter


TRACK_ID = "quadruple-regulation-generalist-v2"


@dataclass
class _Env:
    action_space: spaces.Box


class _CaptureAlgorithm:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.action_space = kwargs["env"].action_space


class _DiagnosticReplay:
    buffer_size = 8

    def __init__(self):
        self.observations = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
        self.actions = np.arange(4, dtype=np.float32).reshape(4, 1, 1)

    def size(self):
        return 4


class _DiagnosticPolicy:
    @staticmethod
    def obs_to_tensor(observations):
        import torch

        return torch.as_tensor(observations), False


class _DiagnosticCritic:
    @staticmethod
    def __call__(observations, actions):
        first = observations.sum(dim=1, keepdim=True) + actions
        return first, 0.5 * first


class _DiagnosticModel:
    _n_updates = 17
    replay_buffer = _DiagnosticReplay()
    policy = _DiagnosticPolicy()
    critic = _DiagnosticCritic()

    class logger:
        name_to_value = {
            "train/actor_loss": -2.0,
            "train/critic_loss": 3.0,
            "train/ent_coef": 0.25,
            "train/n_updates": 17,
        }


def _config(algorithm_id: str, *, algorithm=None) -> RLTrainingConfig:
    return resolve_training_defaults(
        RLTrainingConfig.from_mapping(
            {
                "schema_version": "aiogym.rl_training_config.v3",
                "track_id": TRACK_ID,
                "algorithm_id": algorithm_id,
                "training_seed": 7,
                "budget": {
                    "unit": "environment_transitions",
                    "value": 100,
                },
                "n_envs": 2,
                "algorithm": algorithm or {},
                "validation_seeds": [5000],
            }
        )
    )


def _env(shape=(3,)) -> _Env:
    return _Env(
        spaces.Box(
            low=-np.ones(shape, dtype=np.float32),
            high=np.ones(shape, dtype=np.float32),
            dtype=np.float32,
        )
    )


def test_sac_constructor_receives_frozen_explicit_kwargs(monkeypatch):
    import stable_baselines3
    from torch import nn

    monkeypatch.setattr(stable_baselines3, "SAC", _CaptureAlgorithm)
    model = get_algorithm_adapter("sac").build(_config("sac"), _env())
    kwargs = model.kwargs

    assert kwargs["ent_coef"] == "auto"
    assert kwargs["target_entropy"] == "auto"
    assert kwargs["policy_kwargs"] == {
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [256, 256], "qf": [256, 256]},
    }
    assert "action_noise" not in kwargs
    assert "n_epochs" not in kwargs


def test_td3_constructor_receives_noise_with_action_shape(monkeypatch):
    import stable_baselines3
    from torch import nn

    monkeypatch.setattr(stable_baselines3, "TD3", _CaptureAlgorithm)
    model = get_algorithm_adapter("td3").build(
        _config("td3"),
        _env((2, 3)),
    )
    kwargs = model.kwargs

    assert kwargs["policy_delay"] == 2
    assert kwargs["target_policy_noise"] == pytest.approx(0.2)
    assert kwargs["target_noise_clip"] == pytest.approx(0.5)
    assert kwargs["action_noise"]._mu.shape == (2, 3)
    assert kwargs["action_noise"]._sigma == pytest.approx(
        np.full((2, 3), 0.1)
    )
    assert kwargs["policy_kwargs"]["activation_fn"] is nn.ReLU
    assert "ent_coef" not in kwargs


def test_ppo_constructor_receives_frozen_explicit_kwargs(monkeypatch):
    import stable_baselines3
    from torch import nn

    monkeypatch.setattr(stable_baselines3, "PPO", _CaptureAlgorithm)
    model = get_algorithm_adapter("ppo").build(_config("ppo"), _env())
    kwargs = model.kwargs

    assert kwargs["gae_lambda"] == pytest.approx(0.95)
    assert kwargs["clip_range"] == pytest.approx(0.2)
    assert kwargs["n_epochs"] == 10
    assert kwargs["ent_coef"] == pytest.approx(0.0)
    assert kwargs["vf_coef"] == pytest.approx(0.5)
    assert kwargs["max_grad_norm"] == pytest.approx(0.5)
    assert kwargs["policy_kwargs"] == {
        "activation_fn": nn.Tanh,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }
    assert "train_freq" not in kwargs
    assert "action_noise" not in kwargs


@pytest.mark.parametrize("algorithm_id", ("sac", "td3", "ppo"))
def test_legacy_v3_shape_materializes_policy_defaults(algorithm_id):
    config = _config(algorithm_id)
    policy = config.as_dict()["algorithm"]["policy_kwargs"]

    assert policy["net_arch"]["pi"] == [256, 256]
    assert policy["activation_fn"] == (
        "tanh" if algorithm_id == "ppo" else "relu"
    )


def test_effective_hyperparameter_changes_resolved_config_hash():
    baseline = _config("sac")
    changed = _config("sac", algorithm={"ent_coef": 0.05})

    assert baseline.config_hash != changed.config_hash
    assert changed.as_dict()["algorithm"]["ent_coef"] == pytest.approx(0.05)


def test_training_diagnostics_are_deterministic_and_rng_neutral():
    np.random.seed(123)
    first = _sb3_training_diagnostics(_DiagnosticModel())
    actual_next = np.random.random()
    np.random.seed(123)
    expected_next = np.random.random()
    second = _sb3_training_diagnostics(_DiagnosticModel())

    assert first == second
    assert actual_next == expected_next
    assert first["optimizer_updates"] == 17
    assert first["logger"] == {
        "actor_loss": -2.0,
        "critic_loss": 3.0,
        "ent_coef": 0.25,
        "n_updates": 17.0,
    }
    assert first["replay"] == {"size": 4, "capacity": 8}
    assert first["critic_q"]["sample_count"] == 4
    assert first["critic_q"]["head_count"] == 2
    assert first["critic_q"]["abs_max"] > 0.0


@pytest.mark.parametrize(
    "policy_kwargs, message",
    (
        (
            {
                "activation_fn": "swish",
                "net_arch": {"pi": [64], "qf": [64]},
            },
            "activation_fn",
        ),
        (
            {
                "activation_fn": "relu",
                "net_arch": {"pi": [64], "qf": [0]},
            },
            "positive integers",
        ),
        (
            {
                "activation_fn": "relu",
                "net_arch": {"pi": [64], "vf": [64]},
            },
            "requires exactly",
        ),
        (
            {
                "activation_fn": "relu",
                "net_arch": {"pi": [64], "qf": [64]},
                "optimizer": "adam",
            },
            "unknown algorithm.policy_kwargs",
        ),
    ),
)
def test_policy_kwargs_reject_unknown_or_invalid_values(
    policy_kwargs,
    message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        _config("sac", algorithm={"policy_kwargs": policy_kwargs})


def test_serialized_effective_kwargs_match_adapter_semantics():
    config = _config("td3")
    kwargs = _effective_algorithm_kwargs(
        config,
        train_freq=1,
        gradient_steps=2,
        action_shape=(7,),
    )

    assert kwargs["policy_kwargs"] == config.as_dict()["algorithm"][
        "policy_kwargs"
    ]
    assert kwargs["action_noise"] == "normal"
    assert kwargs["action_noise_sigma"] == pytest.approx(0.1)
    assert kwargs["action_noise_shape"] == [7]
    assert kwargs["train_freq"] == 1
    assert kwargs["gradient_steps"] == 2

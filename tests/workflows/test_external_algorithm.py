from __future__ import annotations

import json
import platform
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest

import aiogym
from aiogym.rl import algorithms as algorithm_registry


@dataclass
class ConstantModel:
    action: np.ndarray
    env: object | None
    seed: int


class ConstantPolicy:
    def __init__(self, model, checkpoint):
        self.env = model.env
        self._action = model.action
        self._checkpoint = str(checkpoint)

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del observation, context
        return self._action.copy()

    def metadata(self):
        return {
            "id": "external_constant",
            "kind": "learned_policy",
            "algorithm": "external_constant",
            "checkpoint": self._checkpoint,
        }


class ConstantAlgorithmBackend:
    id = "external_constant"
    behavior_cloning = None
    requires_dataset = False

    def effective_kwargs(self, *, steps, values):
        del steps
        if values:
            raise ValueError("external_constant accepts no algorithm kwargs")
        return {}

    def create(self, *, env, seed, algorithm_kwargs):
        if algorithm_kwargs:
            raise ValueError("external_constant accepts no algorithm kwargs")
        action = np.asarray(env.unwrapped.model.default_action(), dtype=np.float32)
        return ConstantModel(action=action, env=env, seed=seed)

    def learn(self, model, *, steps, dataset, on_step):
        del dataset
        observation, _ = model.env.reset(seed=model.seed)
        del observation
        episode = 0
        for step in range(1, steps + 1):
            _, reward, terminated, truncated, _ = model.env.step(model.action)
            on_step(
                aiogym.TrainingStep(
                    step=step,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            if terminated or truncated:
                episode += 1
                model.env.reset(seed=model.seed + episode)
        return steps

    def save(self, model, payload):
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "model.json",
                json.dumps({"action": model.action.tolist(), "seed": model.seed}),
            )

    def load(self, payload, *, env=None):
        with zipfile.ZipFile(payload) as archive:
            payload = json.loads(archive.read("model.json"))
        return ConstantModel(
            action=np.asarray(payload["action"], dtype=np.float32),
            env=env,
            seed=int(payload["seed"]),
        )

    def policy(self, model, *, checkpoint):
        return ConstantPolicy(model, checkpoint)

    def runtime_metadata(self):
        return {
            "python": platform.python_version(),
            "aiogym": aiogym.__version__,
            "external_backend": "test_constant.v1",
        }


def test_external_algorithm_uses_the_complete_workflow(tmp_path, capsys):
    backend = ConstantAlgorithmBackend()
    aiogym.register_algorithm(backend)
    env = aiogym.make_env("quadruple")
    evaluation_env = aiogym.make_env("quadruple", randomize=True)
    try:
        assert isinstance(backend, aiogym.AlgorithmBackend)
        assert "external_constant" in aiogym.list_algorithms()

        result = aiogym.train(
            env=env,
            algorithm="external_constant",
            steps=3,
            record_every=2,
            evaluation_env=evaluation_env,
            evaluate_every=2,
            output=tmp_path / "training",
        )
        policy = aiogym.load_policy(
            tmp_path / "training" / "model.zip",
            env=env,
        )
        evaluation = aiogym.evaluate(
            env=env,
            policy=policy,
            seeds=[1],
            max_steps=2,
        )
        comparison = aiogym.compare_policies(
            env=env,
            policies={"external": policy, "hold": "hold"},
            seeds=[1],
            max_steps=2,
            output=tmp_path / "comparison",
        )

        from aiogym.cli.main import main

        assert main(["list", "algorithms"]) == 0
        listed = capsys.readouterr().out.splitlines()
        with zipfile.ZipFile(tmp_path / "training" / "model.zip") as checkpoint:
            manifest = json.loads(checkpoint.read("manifest.json"))
            members = set(checkpoint.namelist())
    finally:
        evaluation_env.close()
        env.close()
        del algorithm_registry._BACKENDS[backend.id]

    assert result["algorithm"] == "external_constant"
    assert result["actual_steps"] == 3
    assert result["runtime"]["external_backend"] == "test_constant.v1"
    assert result["best_checkpoint"].endswith("best/model.zip")
    assert result["evaluation"]["seeds"] == list(range(1_000, 1_020))
    assert evaluation["policy"]["algorithm"] == "external_constant"
    assert set(comparison["evaluations"]) == {"external", "hold"}
    assert "external_constant" in listed
    assert members == {"manifest.json", "payload.zip"}
    assert manifest["schema_version"] == "aiogym.checkpoint.v2"
    assert manifest["algorithm"] == "external_constant"
    assert manifest["environment"]["scenario"] == "quadruple"


def test_load_policy_rejects_non_aiogym_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"not a checkpoint")
    env = aiogym.make_env("quadruple")
    try:
        with pytest.raises(ValueError, match="valid AIO-Gym"):
            aiogym.load_policy(checkpoint, env=env)
    finally:
        env.close()


def test_external_algorithm_checkpoint_can_continue_training(tmp_path):
    backend = ConstantAlgorithmBackend()
    aiogym.register_algorithm(backend)
    env = aiogym.make_env("quadruple")
    try:
        first = aiogym.train(
            env=env,
            algorithm=backend.id,
            steps=2,
            seed=8,
            output=tmp_path / "first",
        )
        continued = aiogym.train(
            env=env,
            algorithm=backend.id,
            steps=3,
            resume_from=first["checkpoint"],
            output=tmp_path / "continued",
        )
        with zipfile.ZipFile(continued["checkpoint"]) as checkpoint:
            manifest = json.loads(checkpoint.read("manifest.json"))
    finally:
        env.close()
        del algorithm_registry._BACKENDS[backend.id]

    assert continued["initial_steps"] == 2
    assert continued["added_steps"] == 3
    assert continued["actual_steps"] == 5
    assert continued["seed"] == 8
    assert manifest["policy"]["training"]["completed_steps"] == 5


def test_installed_entry_point_discovers_external_backend(monkeypatch):
    backend = ConstantAlgorithmBackend()

    class EntryPoint:
        name = backend.id

        @staticmethod
        def load():
            return backend

    monkeypatch.setattr(
        algorithm_registry,
        "entry_points",
        lambda *, group: (EntryPoint(),) if group == "aiogym.algorithms" else (),
    )
    monkeypatch.setattr(
        algorithm_registry,
        "_INSTALLED_BACKENDS_REGISTERED",
        False,
    )
    try:
        assert backend.id in algorithm_registry.list_algorithms()
    finally:
        del algorithm_registry._BACKENDS[backend.id]

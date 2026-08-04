from __future__ import annotations

import hashlib
import json

import pytest

from aiogym._internal.serialization import (
    atomic_write_json,
    canonical_json_bytes,
    file_sha256,
    stable_json_hash,
)
from aiogym.cli.final_test import _stable_hash
from aiogym.generation.registry import load_distribution
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.run_claim import SeedSweepIdentity


def test_canonical_serialization_and_file_hash_contract(tmp_path):
    value = {"unicode": "温度", "finite": 1.25}
    expected = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert canonical_json_bytes(value) == expected
    assert stable_json_hash(value) == hashlib.sha256(expected).hexdigest()
    path = tmp_path / "value.bin"
    path.write_bytes(expected)
    assert file_sha256(path) == hashlib.sha256(expected).hexdigest()


def test_atomic_json_does_not_overwrite_without_permission(tmp_path):
    path = atomic_write_json(tmp_path / "value.json", {"a": 1})
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"a": 2})
    atomic_write_json(path, {"a": 2}, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}


def test_stable_hash_goldens_cover_key_identity_families():
    distribution = load_distribution("quadruple-regulation-training-l2-v1")
    assert distribution.distribution_hash == (
        "e8eb1b1590d62d7b24ab7f6b0ec03f6c5911d7bb0d271eda47f562b8d88bc42a"
    )
    config = RLTrainingConfig(
        track_id="golden",
        algorithm_id="sac",
        training_seed=3,
        total_transitions=20,
        n_envs=1,
        algorithm={"batch_size": 2},
    )
    assert config.config_hash == (
        "ad272f3e1b4cfdf6962621e0d5e1c1cbaba940c5d100d7e37e7c1b7f548c89ab"
    )
    sweep = SeedSweepIdentity(
        "run", "a" * 64, "track", "b" * 64, "sac", (1, 2, 3)
    )
    assert sweep.sweep_id == (
        "c07f6298b791285c2690a038a509ef9f908a89a3f34c8cc3270bf88bc1be0bb9"
    )
    assert _stable_hash({"b": 2, "a": [1, 3]}) == (
        "41206cfbbd2c91b0c47347e15c006841398a76c34ce20c0ae03c5062a8febaef"
    )

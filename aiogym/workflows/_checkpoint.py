"""The single AIO-Gym checkpoint container."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

from aiogym.core.contracts import policy_metadata, validate_policy
from aiogym.core.io import canonical_json_bytes, jsonable
from aiogym.rl.algorithms import get_algorithm

from ._metadata import environment_metadata, validate_environment_compatibility


CHECKPOINT_SCHEMA_VERSION = "aiogym.checkpoint.v2"
_MANIFEST_NAME = "manifest.json"
_PAYLOAD_NAME = "payload.zip"


def save_checkpoint(
    backend,
    model,
    checkpoint,
    *,
    env,
    training,
    overwrite=False,
) -> Path:
    target = _checkpoint_path(checkpoint)
    if not isinstance(overwrite, bool):
        raise TypeError("checkpoint overwrite must be bool")
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=target.parent,
        prefix=f".{target.name}.",
    ) as temporary_name:
        temporary = Path(temporary_name)
        payload = temporary / _PAYLOAD_NAME
        backend.save(model, payload)
        if not payload.is_file():
            raise FileNotFoundError(
                f"algorithm backend {backend.id} did not create payload: {payload}"
            )
        policy = policy_metadata(backend.policy(model, checkpoint=target))
        policy["training"] = _validated_training_state(training)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "algorithm": backend.id,
            "runtime": backend_runtime_metadata(backend),
            "environment": environment_metadata(env),
            "policy": policy,
        }
        archive_path = temporary / target.name
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(_MANIFEST_NAME, canonical_json_bytes(manifest))
            archive.write(payload, arcname=_PAYLOAD_NAME)
        with archive_path.open("rb") as stream:
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
        os.replace(archive_path, target)
    return target


def load_checkpoint(checkpoint, *, env):
    path, _manifest, backend, model = _load_checkpoint_model(checkpoint, env=env)
    return validate_policy(backend.policy(model, checkpoint=path))


def load_training_checkpoint(checkpoint, *, env, algorithm):
    path, manifest, backend, model = _load_checkpoint_model(
        checkpoint,
        env=env,
        training_algorithm=algorithm,
    )
    training = manifest["policy"].get("training")
    if training is None:
        raise ValueError("checkpoint does not contain training state")
    return path, backend, model, _validated_training_state(training)


def _load_checkpoint_model(checkpoint, *, env, training_algorithm=None):
    path = _checkpoint_path(checkpoint)
    if not path.is_file():
        raise ValueError("checkpoint must be an existing model.zip file")
    with tempfile.TemporaryDirectory(prefix="aiogym-checkpoint-") as temporary_name:
        payload = Path(temporary_name) / _PAYLOAD_NAME
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if len(names) != 2 or set(names) != {
                    _MANIFEST_NAME,
                    _PAYLOAD_NAME,
                }:
                    raise ValueError(
                        "checkpoint must contain exactly manifest.json and payload.zip"
                    )
                manifest = _read_manifest(archive)
                validate_environment_compatibility(manifest["environment"], env)
                if training_algorithm is not None:
                    if manifest["algorithm"] != training_algorithm:
                        raise ValueError(
                            f"checkpoint algorithm {manifest['algorithm']!r} does "
                            "not match requested algorithm "
                            f"{training_algorithm!r}"
                        )
                    _validate_training_environment(manifest["environment"], env)
                with archive.open(_PAYLOAD_NAME) as source, payload.open(
                    "wb"
                ) as destination:
                    shutil.copyfileobj(source, destination)
        except zipfile.BadZipFile as error:
            raise ValueError("checkpoint must be a valid AIO-Gym model.zip") from error
        backend = get_algorithm(manifest["algorithm"])
        model = backend.load(payload, env=env)
    return path, manifest, backend, model


def _validate_training_environment(expected, env) -> None:
    actual = environment_metadata(env)
    if set(expected) != set(actual):
        raise ValueError("checkpoint training environment fields are invalid")
    for field, value in expected.items():
        if field not in actual or value != actual[field]:
            raise ValueError(
                f"checkpoint {field} is incompatible with the training environment"
            )


def backend_runtime_metadata(backend) -> dict:
    runtime = backend.runtime_metadata()
    if not isinstance(runtime, Mapping):
        raise TypeError("algorithm backend runtime_metadata must return a mapping")
    serialized = jsonable(dict(runtime))
    if not isinstance(serialized, dict):
        raise TypeError("algorithm backend runtime_metadata must return a mapping")
    return serialized


def _checkpoint_path(checkpoint) -> Path:
    path = Path(checkpoint)
    if path.name != "model.zip":
        raise ValueError("checkpoint path must end with model.zip")
    return path


def _read_manifest(archive) -> dict:
    try:
        manifest = json.loads(archive.read(_MANIFEST_NAME))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint manifest.json must be valid JSON") from error
    expected = {
        "schema_version",
        "algorithm",
        "runtime",
        "environment",
        "policy",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError(
            "checkpoint manifest requires schema_version, algorithm, runtime, "
            "environment, and policy"
        )
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version must be {CHECKPOINT_SCHEMA_VERSION}"
        )
    if not isinstance(manifest["algorithm"], str) or not manifest["algorithm"]:
        raise TypeError("checkpoint algorithm must be a non-empty string")
    for field in ("runtime", "environment", "policy"):
        if not isinstance(manifest[field], dict):
            raise TypeError(f"checkpoint {field} must be a JSON object")
    return manifest


def _validated_training_state(value) -> dict:
    serialized = jsonable(value)
    expected = {"completed_steps", "seed", "algorithm_kwargs", "dataset"}
    if not isinstance(serialized, dict) or set(serialized) != expected:
        raise ValueError(
            "checkpoint training state requires completed_steps, seed, "
            "algorithm_kwargs, and dataset"
        )
    for name in ("completed_steps", "seed"):
        item = serialized[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"checkpoint training {name} must be non-negative")
    if not isinstance(serialized["algorithm_kwargs"], dict):
        raise TypeError("checkpoint training algorithm_kwargs must be a JSON object")
    if serialized["dataset"] is not None and not isinstance(
        serialized["dataset"], dict
    ):
        raise TypeError("checkpoint training dataset must be a JSON object or null")
    return serialized


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "backend_runtime_metadata",
    "load_checkpoint",
    "load_training_checkpoint",
    "save_checkpoint",
]

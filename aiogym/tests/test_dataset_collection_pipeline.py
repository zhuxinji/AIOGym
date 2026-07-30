from __future__ import annotations

import json

from aiogym.datasets import DatasetCollectionConfig, DatasetReader
from aiogym.datasets.collect import collect_dataset
from aiogym.rl.dataset_replay import DatasetReplay


def _config(output, **overrides):
    declaration = {
        "schema_version": "aiogym.dataset_collection.v1",
        "track_id": "quadruple-regulation-generalist-v1",
        "dataset_id": "collection-pipeline-v1",
        "split": "training",
        "base_seed": 17,
        "target_transitions": 20,
        "workers": 1,
        "collectors": [
            {"id": "nominal_pid", "weight": 0.4},
            {"id": "noisy_pid", "weight": 0.6},
        ],
        "output": str(output),
    }
    declaration.update(overrides)
    return DatasetCollectionConfig(declaration)


def test_config_collection_budget_manifest_and_resume(tmp_path):
    config = _config(tmp_path / "dataset")
    first = collect_dataset(config)
    assert first["actual_transitions"] >= first["requested_transitions"]
    reader = DatasetReader(config.output)
    collection = reader.manifest["collection"]
    assert collection["config_hash"] == config.config_hash
    assert collection["requested_transitions"] == 20
    assert collection["actual_transitions"] == reader.transition_count
    assert collection["complete_episode_overshoot"] is True
    assert collection["completed"] is True
    assert [
        row["episode_index"] for row in reader.metadata_records()
    ] == list(range(len(reader)))

    second = collect_dataset(config, resume=True)
    assert second["episodes"] == first["episodes"]
    assert DatasetReader(config.output).episode_ids == reader.episode_ids


def test_reader_checksum_mode_is_cached_and_optional(tmp_path, monkeypatch):
    config = _config(tmp_path / "checksum")
    collect_dataset(config)
    calls = []

    from aiogym.datasets import reader as reader_module

    original = reader_module.file_sha256

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(reader_module, "file_sha256", counted)
    unchecked = DatasetReader(config.output, verify_checksums=False)
    unchecked.load_episode(0)
    unchecked.load_episode(0)
    assert calls == []

    checked = DatasetReader(config.output, verify_checksums=True)
    initial_calls = len(calls)
    assert initial_calls == len(checked)
    checked.load_episode(0)
    checked.load_episode(0)
    assert len(calls) == initial_calls


def test_replay_precomputes_sampling_tables_and_batches_by_episode(
    tmp_path,
):
    config = _config(tmp_path / "replay")
    collect_dataset(config)
    replay = DatasetReplay(config.output, seed=4)
    batch = replay.sample(32)
    assert batch["observation"].shape[0] == 32
    assert batch["action_policy_normalized"].shape[0] == 32
    assert batch["source"].tolist() == ["offline"] * 32
    assert replay._sampling


def test_top_level_collect_cli_accepts_config_file(tmp_path, capsys):
    from aiogym.cli.main import main

    config = _config(tmp_path / "cli")
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    assert main(["collect", "--config", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_id"] == config.dataset_id
    assert payload["actual_transitions"] >= 20

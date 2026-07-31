from __future__ import annotations

from aiogym.datasets import DatasetCollectionConfig, DatasetReader
from aiogym.datasets.collect import collect_dataset


def _config(output, workers):
    return DatasetCollectionConfig(
        {
            "schema_version": "aiogym.dataset_collection.v1",
            "track_id": "quadruple-regulation-generalist-v1",
            "dataset_id": "parallel-identity-v1",
            "split": "training",
            "base_seed": 23,
            "target_transitions": 1000,
            "workers": workers,
            "collectors": [
                {"id": "nominal_pid", "weight": 0.5},
                {"id": "noisy_pid", "weight": 0.5},
            ],
            "output": str(output),
        }
    )


def test_worker_count_does_not_change_episode_or_collector_identities(
    tmp_path,
):
    serial = _config(tmp_path / "serial", 1)
    parallel = _config(tmp_path / "parallel", 2)
    collect_dataset(serial)
    collect_dataset(parallel)

    def identities(path):
        return {
            (
                record["episode_index"],
                record["episode_spec_id"],
                record["resolved_hash"],
                record["collector_id"],
                record["content_hash"],
            )
            for record in DatasetReader(path).metadata_records()
        }

    assert identities(serial.output) == identities(parallel.output)
    parallel_records = DatasetReader(parallel.output).metadata_records()
    assert [row["episode_index"] for row in parallel_records] == [0, 1]
    for record in parallel_records:
        metadata = record["episode"]["metadata"]
        assert "requested_initial_state" in metadata
        assert "reset_state_delta_linf" in metadata


def test_resume_continues_after_max_committed_episode_index(tmp_path):
    output = tmp_path / "resume"
    config = _config(output, 1)
    collect_dataset(config)
    before = DatasetReader(output)
    ids = before.episode_ids
    result = collect_dataset(config, resume=True)
    after = DatasetReader(output)
    assert after.episode_ids == ids
    assert result["episodes"] == len(ids)
    assert len(set(after.episode_ids)) == len(after)

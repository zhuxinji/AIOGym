from __future__ import annotations

import json

import numpy as np
import pytest

from aiogym.datasets.collector import collect_episode, list_collectors
from aiogym.datasets.minari_adapter import (
    episode_from_minari_dict,
    episode_to_minari_dict,
)
from aiogym.datasets.quality import build_quality_report, write_quality_report
from aiogym.datasets.reader import DatasetReader, validate_dataset
from aiogym.datasets.schema import DATASET_SCHEMA_VERSION, DatasetEpisode
from aiogym.datasets.writer import DatasetWriter
from aiogym.tests._env import make_test_env as make_env
from aiogym.benchmarks import load_track
from aiogym.generation.cascade import CascadeTrainingSampler
from aiogym.generation.quadruple import (
    QuadrupleTrainingSampler,
)
from aiogym.rl.episode_env import make_track_training_base_env


def test_dataset_episode_captures_v2_action_and_bootstrap_semantics():
    episode = _collected_episode(1)
    assert episode.transition_count == 12
    assert episode.array("observation").shape[0] == 12
    assert episode.array("true_state").shape == (12, 4)
    assert episode.array("action_policy_normalized").shape == (12, 2)
    assert episode.array("action_commanded_physical").shape == (12, 2)
    assert episode.array("action_applied_physical").shape == (12, 2)
    assert np.all(
        np.abs(episode.array("action_policy_normalized")) <= 1.0
    )
    assert episode.array("truncated")[-1]
    assert not episode.array("terminated")[-1]
    assert episode.array("bootstrap_mask")[-1] == 1.0
    assert episode.reward_term_names
    assert episode.cost_channel_names
    _assert_reset_state_metadata(episode)


def test_cascade_temperature_initial_state_uses_recorded_reset_snapshot():
    track = load_track("cascade-regulation-generalist-v1")
    sampler = CascadeTrainingSampler(track.training_distribution())
    episode_spec = sampler.sample(408)
    env = make_track_training_base_env(track, sampler=sampler)
    try:
        episode = collect_episode(
            env,
            episode_spec,
            collector_id="safe_excitation",
            track_id=track.id,
            max_steps=1,
        )
    finally:
        env.close()

    _assert_reset_state_metadata(episode)
    metadata = episode.metadata
    requested = np.asarray(
        metadata["requested_initial_state"],
        dtype=np.float64,
    )
    recorded = np.asarray(
        metadata["initial_state"],
        dtype=np.float64,
    )
    assert metadata["reset_state_delta_linf"] > 1e-6
    assert int(np.argmax(np.abs(requested - recorded))) in {1, 3, 5}


def test_dataset_v2_allows_omitting_optional_reset_diagnostics():
    source = _collected_episode(16)
    metadata = source.metadata
    metadata.pop("requested_initial_state")
    metadata.pop("reset_state_delta_linf")

    without_diagnostics = _rebuild(source, metadata=metadata)
    restored = DatasetEpisode.from_storage(
        without_diagnostics.storage_metadata(),
        without_diagnostics.arrays(),
    )

    assert restored.metadata == metadata
    assert restored.content_hash == without_diagnostics.content_hash
    assert without_diagnostics.storage_metadata()["schema_version"] == (
        "aiogym.dataset.episode.v2"
    )
    assert np.array_equal(
        np.asarray(
            without_diagnostics.metadata["initial_state"],
            dtype=without_diagnostics.array("true_state").dtype,
        ),
        without_diagnostics.array("true_state")[0],
    )


def test_dataset_v2_rejects_real_initial_state_mismatch():
    source = _collected_episode(17)
    metadata = source.metadata
    metadata["initial_state"][0] += 1e-3
    metadata["reset_state_delta_linf"] = float(
        np.max(
            np.abs(
                np.asarray(
                    metadata["requested_initial_state"],
                    dtype=np.float64,
                )
                - np.asarray(metadata["initial_state"], dtype=np.float64)
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="metadata initial_state must match the first true_state",
    ):
        _rebuild(source, metadata=metadata)


def test_dataset_v2_validates_optional_reset_diagnostics_as_a_pair():
    source = _collected_episode(18)
    metadata = source.metadata
    metadata.pop("reset_state_delta_linf")
    with pytest.raises(ValueError, match="must be provided together"):
        _rebuild(source, metadata=metadata)

    metadata = source.metadata
    metadata["reset_state_delta_linf"] += 1.0
    with pytest.raises(ValueError, match="does not match"):
        _rebuild(source, metadata=metadata)


def test_dataset_round_trip_preserves_episode(tmp_path):
    episode = _collected_episode(2)
    path = tmp_path / "dataset"
    with DatasetWriter(
        path,
        dataset_id="quadruple-smoke-v2",
        split="training",
    ) as writer:
        record = writer.append_episode(episode)

    reader = DatasetReader(path, verify_checksums=True)
    restored = reader.load_episode(episode.episode_id)
    assert restored.content_hash == episode.content_hash
    assert restored.metadata == episode.metadata
    for name, array in episode.arrays().items():
        assert np.array_equal(restored.array(name), array)
    assert record["shard_sha256"]
    assert reader.manifest["schema_version"] == DATASET_SCHEMA_VERSION


def test_dataset_writer_resumes_without_duplicates(tmp_path):
    first = _collected_episode(3)
    second = _collected_episode(4)
    path = tmp_path / "resume"
    with DatasetWriter(
        path,
        dataset_id="resume-v2",
        split="training",
    ) as writer:
        writer.append_episode(first)
    with DatasetWriter(
        path,
        dataset_id="resume-v2",
        split="training",
        resume=True,
    ) as writer:
        with pytest.raises(ValueError, match="duplicate episode_id"):
            writer.append_episode(first)
        writer.append_episode(second)

    reader = DatasetReader(path)
    assert reader.episode_ids == (first.episode_id, second.episode_id)
    assert reader.transition_count == 24
    assert reader.validate_integrity()["ok"]


def test_manifest_failure_removes_newly_written_shard(
    tmp_path, monkeypatch
):
    path = tmp_path / "manifest-failure"
    writer = DatasetWriter(
        path,
        dataset_id="manifest-failure-v2",
        split="training",
    )
    monkeypatch.setattr(
        writer,
        "_write_manifest",
        lambda: (_ for _ in ()).throw(RuntimeError("manifest commit failed")),
    )
    with pytest.raises(RuntimeError, match="manifest commit failed"):
        writer.append_episode(_collected_episode(30))
    assert list((path / "shards").glob("part-*.npz")) == []
    assert writer.manifest["episode_count"] == 0


def test_resume_quarantines_orphan_shard_idempotently(tmp_path):
    path = tmp_path / "orphan"
    with DatasetWriter(
        path,
        dataset_id="orphan-v2",
        split="training",
    ) as writer:
        writer.append_episode(_collected_episode(31))
    orphan = path / "shards" / "part-00000007.npz"
    np.savez_compressed(orphan, orphan=np.asarray([1]))

    with DatasetWriter(
        path,
        dataset_id="orphan-v2",
        split="training",
        resume=True,
    ) as writer:
        events = writer.manifest["collection"]["recovery_events"]
        assert len(events) == 1
        assert events[0]["source"] == "shards/part-00000007.npz"
        writer.append_episode(_collected_episode(32))
    assert (path / "shards" / "part-00000001.npz").is_file()
    quarantined = list((path / "shards" / ".orphaned").iterdir())
    assert len(quarantined) == 1

    with DatasetWriter(
        path,
        dataset_id="orphan-v2",
        split="training",
        resume=True,
    ) as writer:
        assert len(
            writer.manifest["collection"]["recovery_events"]
        ) == 1


def test_missing_referenced_shard_is_not_treated_as_orphan(tmp_path):
    path = tmp_path / "missing"
    with DatasetWriter(
        path,
        dataset_id="missing-v2",
        split="training",
    ) as writer:
        record = writer.append_episode(_collected_episode(33))
    (path / record["shard"]).unlink()
    with pytest.raises(FileNotFoundError, match="referenced shard"):
        DatasetWriter(
            path,
            dataset_id="missing-v2",
            split="training",
            resume=True,
        )


def test_manifest_hash_and_shard_checksum_detect_corruption(tmp_path):
    episode = _collected_episode(5)
    path = tmp_path / "corrupt"
    with DatasetWriter(
        path,
        dataset_id="corrupt-v2",
        split="training",
    ) as writer:
        record = writer.append_episode(episode)
    shard = path / record["shard"]
    with shard.open("r+b") as stream:
        stream.seek(max(0, shard.stat().st_size // 2))
        byte = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([byte[0] ^ 0xFF]))

    report = validate_dataset(path)
    assert not report["ok"]
    assert any("checksum" in error for error in report["errors"])

    manifest = path / "manifest.json"
    data = json.loads(manifest.read_text())
    data["transition_count"] += 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest hash"):
        DatasetReader(path)


def test_split_is_episode_atomic(tmp_path):
    training = _collected_episode(6, split="training")
    validation = _collected_episode(7, split="validation")
    with DatasetWriter(
        tmp_path / "train",
        dataset_id="atomic-v2",
        split="training",
    ) as writer:
        writer.append_episode(training)
        with pytest.raises(ValueError, match="episode split"):
            writer.append_episode(validation)


def test_random_batch_reads_without_loading_dataset_eagerly(tmp_path):
    episodes = [_collected_episode(seed) for seed in (8, 9, 10)]
    path = tmp_path / "batch"
    with DatasetWriter(
        path,
        dataset_id="batch-v2",
        split="training",
    ) as writer:
        for episode in episodes:
            writer.append_episode(episode)
    reader = DatasetReader(path)
    records = reader.metadata_records()
    assert len(records) == 3
    batch = reader.random_batch(17, rng=np.random.default_rng(1))
    assert batch["observation"].shape[0] == 17
    assert batch["action_policy_normalized"].shape == (17, 2)
    assert set(batch["episode_id"]) <= set(reader.episode_ids)


def test_quality_report_contains_coverage_and_provenance(tmp_path):
    path = tmp_path / "quality"
    with DatasetWriter(
        path,
        dataset_id="quality-v2",
        split="training",
    ) as writer:
        writer.append_episode(_collected_episode(11))
        writer.append_episode(_collected_episode(12))
    report = build_quality_report(path)
    assert report["episodes"] == 2
    assert report["transitions"] == 24
    assert report["collector_composition"] == {"safe_excitation": 2}
    assert "normalized_coverage" in report["state"]
    assert "saturation_rate" in report["action"]
    assert report["integrity"]["nonfinite_count"] == 0
    write_quality_report(path)
    assert (path / "reports" / "quality.json").is_file()
    assert (path / "reports" / "coverage.json").is_file()


def test_terminated_truncated_bootstrap_mask_round_trip(tmp_path):
    truncated = _collected_episode(13)
    metadata = truncated.metadata
    metadata["episode_id"] += ":terminal"
    metadata["termination_reason"] = "safety_terminal"
    arrays = truncated.arrays()
    terminated = arrays["terminated"].copy()
    truncated_flags = arrays["truncated"].copy()
    bootstrap = arrays["bootstrap_mask"].copy()
    terminated[-1] = True
    truncated_flags[-1] = False
    bootstrap[-1] = 0.0
    terminal = _rebuild(
        truncated,
        metadata=metadata,
        terminated=terminated,
        truncated=truncated_flags,
        bootstrap_mask=bootstrap,
    )
    path = tmp_path / "termination"
    with DatasetWriter(
        path,
        dataset_id="termination-v2",
        split="training",
    ) as writer:
        writer.append_episode(terminal)
    restored = DatasetReader(path).load_episode(terminal.episode_id)
    assert restored.array("terminated")[-1]
    assert not restored.array("truncated")[-1]
    assert restored.array("bootstrap_mask")[-1] == 0.0


def test_commanded_and_applied_actions_remain_distinct(tmp_path):
    source = _collected_episode(14)
    metadata = source.metadata
    metadata["episode_id"] += ":lagged"
    metadata["actuator_model"] = {
        "kind": "first_order_lag",
        "time_constant": 2.0,
    }
    applied = 0.8 * source.array("action_commanded_physical")
    lagged = _rebuild(
        source,
        metadata=metadata,
        action_applied_physical=applied,
    )
    path = tmp_path / "actions"
    with DatasetWriter(
        path,
        dataset_id="actions-v2",
        split="training",
    ) as writer:
        writer.append_episode(lagged)
    restored = DatasetReader(path).load_episode(lagged.episode_id)
    assert not np.array_equal(
        restored.array("action_commanded_physical"),
        restored.array("action_applied_physical"),
    )


def test_minari_sidecar_round_trip_preserves_boundaries_and_actions():
    episode = _collected_episode(15)
    payload = episode_to_minari_dict(episode)
    restored = episode_from_minari_dict(payload)
    assert restored.content_hash == episode.content_hash
    assert payload["observations"].shape[0] == episode.transition_count + 1
    assert np.array_equal(
        restored.array("terminated"),
        episode.array("terminated"),
    )
    assert np.array_equal(
        restored.array("action_policy_normalized"),
        episode.array("action_policy_normalized"),
    )
    assert (
        restored.metadata["reset_state_delta_linf"]
        == episode.metadata["reset_state_delta_linf"]
    )


def test_collector_registry_has_mixed_policy_sources():
    assert {
        "nominal_pid",
        "parameter_randomized_pid",
        "mpc",
        "noisy_pid",
        "checkpoint",
        "safe_excitation",
        "recovery",
    } <= set(list_collectors())


def _collected_episode(seed, *, split="training"):
    sampler = QuadrupleTrainingSampler(
        level="L2",
        split=split,
        episode_steps=12,
    )
    episode_spec = sampler.sample(seed)
    env = make_env(
        "quadruple",
        reward_spec="regulation-v1",
        control_dt=1.0,
        episode_steps=12,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
        noise=False,
        previous_action_obs=True,
        normalize_observations=True,
        tracking_error_obs=True,
    )
    try:
        return collect_episode(
            env,
            episode_spec,
            collector_id="safe_excitation",
            track_id="quadruple-regulation-generalist-v1",
            split=split,
        )
    finally:
        env.close()


def _rebuild(source, *, metadata, **overrides):
    arrays = source.arrays()
    arrays.update(overrides)
    return DatasetEpisode(
        metadata=metadata,
        observation=arrays["observation"],
        true_state=arrays["true_state"],
        reference=arrays["reference"],
        measured_disturbance=arrays["measured_disturbance"],
        action_policy_normalized=arrays["action_policy_normalized"],
        action_commanded_physical=arrays["action_commanded_physical"],
        action_applied_physical=arrays["action_applied_physical"],
        reward_scalar=arrays["reward_scalar"],
        reward_terms=source.reward_terms,
        cost_channels=source.cost_channels,
        next_observation=arrays["next_observation"],
        next_true_state=arrays["next_true_state"],
        terminated=arrays["terminated"],
        truncated=arrays["truncated"],
        bootstrap_mask=arrays["bootstrap_mask"],
        step_index=arrays["step_index"],
        physical_time=arrays["physical_time"],
    )


def _assert_reset_state_metadata(episode):
    metadata = episode.metadata
    true_state = episode.array("true_state")
    recorded = np.asarray(
        metadata["initial_state"],
        dtype=true_state.dtype,
    )
    requested = np.asarray(
        metadata["requested_initial_state"],
        dtype=np.float64,
    )
    assert np.array_equal(recorded, true_state[0])
    assert metadata["reset_state_delta_linf"] == float(
        np.max(
            np.abs(
                requested - recorded.astype(np.float64)
            )
        )
    )

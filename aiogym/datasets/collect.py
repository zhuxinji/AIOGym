"""Deterministic, parallel Dataset v2 collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor

from aiogym.benchmarks import load_track
from aiogym.generation import make_episode_sampler
from aiogym.rl.episode_env import (
    make_track_episode_sampler,
    make_track_training_base_env,
)

from .collector import collect_episode, get_collector
from .collector_adapters import (
    get_collector_adapter,
    make_collector_behavior,
)
from .config import DatasetCollectionConfig
from .quality import write_quality_report
from .schema import DatasetEpisode
from .writer import DatasetWriter


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Collect a reproducible episode-oriented AIO-Gym Dataset v2."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    config = DatasetCollectionConfig.load(args.config)
    result = collect_dataset(config, resume=bool(args.resume))
    print(json.dumps(result, sort_keys=True))
    return 0


def collect_dataset(
    config: DatasetCollectionConfig | dict,
    *,
    resume: bool = False,
) -> dict:
    """Collect complete episodes and commit them in global-index order."""

    resolved = (
        config
        if isinstance(config, DatasetCollectionConfig)
        else DatasetCollectionConfig(config)
    )
    track, sampler = validate_collection_capabilities(resolved)
    distribution = sampler.distribution
    collection_metadata = {
        "schema_version": resolved.declaration["schema_version"],
        "config_hash": resolved.config_hash,
        "track_id": track.id,
        "track_hash": track.track_hash,
        "distribution_id": distribution.distribution_id,
        "distribution_hash": distribution.distribution_hash,
        "requested_transitions": resolved.target_transitions,
        "actual_transitions": 0,
        "complete_episode_overshoot": True,
        "workers": resolved.workers,
        "collector_weights": [
            allocation.as_dict() for allocation in resolved.collectors
        ],
        "completed": False,
    }
    writer = DatasetWriter(
        resolved.output,
        dataset_id=resolved.dataset_id,
        split=resolved.split,
        resume=resume,
        collection_metadata=collection_metadata,
    )
    manifest = writer.manifest
    committed_indexes = [
        int(record["episode_index"])
        for record in manifest["episodes"]
        if record.get("episode_index") is not None
    ]
    episode_index = max(
        committed_indexes,
        default=int(manifest["episode_count"]) - 1,
    ) + 1
    try:
        while (
            writer.manifest["transition_count"]
            < resolved.target_transitions
        ):
            remaining = (
                resolved.target_transitions
                - writer.manifest["transition_count"]
            )
            estimated_episodes = max(
                1,
                math.ceil(remaining / distribution.episode_steps),
            )
            batch_size = min(resolved.workers, estimated_episodes)
            indexes = tuple(
                range(episode_index, episode_index + batch_size)
            )
            episodes = _collect_index_batch(resolved, indexes)
            for expected_index, episode in zip(indexes, episodes):
                actual_index = int(
                    episode.metadata.get("episode_index", -1)
                )
                if actual_index != expected_index:
                    raise RuntimeError(
                        "worker returned a non-canonical episode index"
                    )
                writer.append_episode(episode)
                writer.update_collection_metadata(
                    actual_transitions=writer.manifest[
                        "transition_count"
                    ],
                    next_episode_index=expected_index + 1,
                    completed=(
                        writer.manifest["transition_count"]
                        >= resolved.target_transitions
                    ),
                )
            episode_index += batch_size
    finally:
        writer.close()
    report = write_quality_report(resolved.output)
    return {
        "dataset_id": resolved.dataset_id,
        "path": str(resolved.output),
        "config_hash": resolved.config_hash,
        "episodes": report["episodes"],
        "requested_transitions": resolved.target_transitions,
        "actual_transitions": report["transitions"],
        "workers": resolved.workers,
    }


def validate_collection_capabilities(
    config: DatasetCollectionConfig,
):
    """Fail before creating output when a collector cannot run on a Track."""

    if not isinstance(config, DatasetCollectionConfig):
        raise TypeError("config must be a DatasetCollectionConfig")
    track = load_track(config.track_id)
    sampler = make_track_episode_sampler(track)
    episode = sampler.sample(config.base_seed, episode_index=0)
    for allocation in config.collectors:
        get_collector(allocation.collector_id)
        get_collector_adapter(allocation.collector_id)
        env = make_track_training_base_env(track, sampler=sampler)
        behavior = None
        try:
            behavior = make_collector_behavior(
                allocation.collector_id,
                env,
                episode,
                options=allocation.options,
            )
        except Exception as exc:
            raise RuntimeError(
                f"collector {allocation.collector_id!r} is not "
                f"supported for Track {track.id!r}: {exc}"
            ) from exc
        finally:
            target = behavior.env if behavior is not None else env
            target.close()
    return track, sampler


def _collect_index_batch(
    config: DatasetCollectionConfig,
    indexes: tuple[int, ...],
) -> list[DatasetEpisode]:
    payloads = [
        {
            "config": config.as_dict(),
            "config_hash": config.config_hash,
            "episode_index": episode_index,
            "collector": _collector_for_index(config, episode_index),
        }
        for episode_index in indexes
    ]
    if config.workers == 1 or len(payloads) == 1:
        serialized = [_collect_job(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(payloads))
        ) as executor:
            serialized = list(executor.map(_collect_job, payloads))
    return [
        DatasetEpisode.from_storage(storage, arrays)
        for storage, arrays in serialized
    ]


def _collect_job(payload):
    config = DatasetCollectionConfig(payload["config"])
    episode_index = int(payload["episode_index"])
    collector = dict(payload["collector"])
    track = load_track(config.track_id)
    if track.train_distribution_id is not None:
        sampler = make_episode_sampler(
            track.training_distribution(),
            split=config.split,
        )
    else:
        sampler = make_track_episode_sampler(track)
    episode_spec = sampler.sample(
        config.base_seed,
        episode_index=episode_index,
    )
    env = make_track_training_base_env(track, sampler=sampler)
    behavior = make_collector_behavior(
        collector["id"],
        env,
        episode_spec,
        options=collector.get("options") or {},
    )
    try:
        episode = collect_episode(
            behavior.env,
            episode_spec,
            collector_id=collector["id"],
            policy=behavior.policy,
            track_id=track.id,
            split=config.split,
            checkpoint_hash=behavior.checkpoint_hash,
            episode_index=episode_index,
        )
        arrays = {
            name: value.copy()
            for name, value in episode.arrays().items()
        }
        return episode.storage_metadata(), arrays
    finally:
        behavior.env.close()


def _collector_for_index(
    config: DatasetCollectionConfig,
    episode_index: int,
) -> dict:
    allocations = config.collectors
    total = sum(row.weight for row in allocations)
    digest = hashlib.sha256(
        (
            f"{_identity_namespace(config)}:{config.base_seed}:"
            f"{int(episode_index)}"
        ).encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64)
    threshold = draw * total
    cumulative = 0.0
    for allocation in allocations:
        cumulative += allocation.weight
        if threshold < cumulative:
            return allocation.as_dict()
    return allocations[-1].as_dict()


def _identity_namespace(config: DatasetCollectionConfig) -> str:
    payload = {
        "schema_version": config.declaration["schema_version"],
        "track_id": config.track_id,
        "dataset_id": config.dataset_id,
        "split": config.split,
        "base_seed": config.base_seed,
        "collectors": [
            allocation.as_dict() for allocation in config.collectors
        ],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "collect_dataset",
    "main",
    "validate_collection_capabilities",
]

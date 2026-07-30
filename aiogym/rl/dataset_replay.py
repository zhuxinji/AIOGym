"""Random-access and source-stratified replay over Dataset v2."""
from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from aiogym.datasets import DatasetReader
from aiogym.datasets.writer import manifest_with_hash


class DatasetReplay:
    """Immutable offline replay backed by episode shards.

    Sampling is uniform over source strata, then transition-weighted within a
    stratum. This prevents a large collector from silently drowning out small
    recovery or expert sources.
    """

    FIELDS = (
        "observation",
        "action_policy_normalized",
        "reward_scalar",
        "next_observation",
        "terminated",
        "truncated",
        "bootstrap_mask",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        seed: int = 0,
        stratify: bool = True,
        cache_episodes: int = 8,
        verify_checksums: bool = False,
    ) -> None:
        self.path = Path(path)
        self.reader = DatasetReader(
            self.path,
            verify_checksums=verify_checksums,
        )
        if self.reader.transition_count <= 0:
            raise ValueError("offline dataset must contain transitions")
        self.dataset_id = str(self.reader.manifest["dataset_id"])
        self.dataset_hash = manifest_with_hash(
            self.reader.manifest
        )["manifest_hash"]
        self.stratify = bool(stratify)
        self.cache_episodes = max(1, int(cache_episodes))
        self._rng = np.random.default_rng(int(seed))
        self.sample_count = 0
        self._records = self.reader.metadata_records()
        self._cache: OrderedDict[int, object] = OrderedDict()
        grouped = defaultdict(list)
        for index, record in enumerate(self._records):
            grouped[self._stratum(record)].append(index)
        self._strata = {
            stratum: tuple(indices)
            for stratum, indices in sorted(grouped.items())
        }
        self._sampling = {
            stratum: self._sampling_table(indices)
            for stratum, indices in self._strata.items()
        }
        self._all_sampling = self._sampling_table(
            tuple(range(len(self._records)))
        )
        first = self.reader.load_episode(0)
        self.observation_dim = int(first.array("observation").shape[1])
        self.action_dim = int(
            first.array("action_policy_normalized").shape[1]
        )

    def __len__(self) -> int:
        return self.reader.transition_count

    @property
    def strata(self) -> tuple[str, ...]:
        return tuple(self._strata)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        size = int(batch_size)
        rows = {name: [None] * size for name in self.FIELDS}
        episode_ids = [None] * size
        collector_ids = [None] * size
        difficulty_tags = [None] * size
        strata = self.strata
        if self.stratify:
            selected_strata = np.asarray(
                [
                    strata[index]
                    for index in self._rng.integers(
                        0,
                        len(strata),
                        size=size,
                    )
                ],
                dtype=np.str_,
            )
        else:
            selected_strata = np.full(size, "all", dtype=np.str_)
        episode_indices = np.empty(size, dtype=np.int64)
        for stratum in sorted(set(selected_strata.tolist())):
            positions = np.flatnonzero(selected_strata == stratum)
            candidates, probabilities = (
                self._all_sampling
                if stratum == "all"
                else self._sampling[stratum]
            )
            episode_indices[positions] = self._rng.choice(
                candidates,
                size=len(positions),
                p=probabilities,
            )
        local_indices = np.asarray(
            [
                int(
                    self._rng.integers(
                        0,
                        self._records[int(episode_index)][
                            "transition_count"
                        ],
                    )
                )
                for episode_index in episode_indices
            ],
            dtype=np.int64,
        )
        for episode_index in sorted(set(episode_indices.tolist())):
            positions = np.flatnonzero(episode_indices == episode_index)
            record = self._records[int(episode_index)]
            episode = self._episode(int(episode_index))
            for offset in positions:
                local_index = int(local_indices[offset])
                for name in self.FIELDS:
                    rows[name][offset] = episode.array(name)[local_index]
                episode_ids[offset] = record["episode_id"]
                collector_ids[offset] = record["collector_id"]
                stratum = str(selected_strata[offset])
                difficulty_tags[offset] = (
                    stratum.split("|", 1)[1]
                    if "|" in stratum
                    else "all"
                )
        self.sample_count += size
        result = {
            name: np.asarray(values) for name, values in rows.items()
        }
        result.update(
            {
                "source": np.full(batch_size, "offline", dtype="<U16"),
                "episode_id": np.asarray(episode_ids, dtype=np.str_),
                "collector_id": np.asarray(
                    collector_ids,
                    dtype=np.str_,
                ),
                "difficulty_tag": np.asarray(
                    difficulty_tags,
                    dtype=np.str_,
                ),
            }
        )
        return result

    def _sampling_table(self, indices):
        candidates = np.asarray(indices, dtype=np.int64)
        weights = np.asarray(
            [
                self._records[int(index)]["transition_count"]
                for index in candidates
            ],
            dtype=np.float64,
        )
        weights /= np.sum(weights)
        return candidates, weights

    def state_dict(self) -> dict:
        return {
            "path": str(self.path.resolve()),
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "stratify": self.stratify,
            "cache_episodes": self.cache_episodes,
            "sample_count": self.sample_count,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state) -> "DatasetReplay":
        result = cls(
            state["path"],
            stratify=bool(state["stratify"]),
            cache_episodes=int(state["cache_episodes"]),
        )
        if result.dataset_id != state["dataset_id"]:
            raise ValueError("offline replay dataset_id changed")
        if result.dataset_hash != state["dataset_hash"]:
            raise ValueError("offline replay dataset hash changed")
        result.sample_count = int(state["sample_count"])
        result._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return result

    def metadata(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "dataset_path": str(self.path),
            "offline_transitions": len(self),
            "source_stratified": self.stratify,
            "strata": list(self.strata),
        }

    def _stratum(self, record) -> str:
        collector = str(record.get("collector_id") or "unknown")
        tags = tuple(record.get("difficulty_tags") or ())
        tag = str(tags[0]) if tags else "untagged"
        return f"{collector}|{tag}"

    def _episode(self, index: int):
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        value = self.reader.load_episode(index)
        self._cache[index] = value
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return value


__all__ = ["DatasetReplay"]

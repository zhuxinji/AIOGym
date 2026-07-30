"""Optional Minari-compatible episode conversion with an AIO-Gym sidecar."""
from __future__ import annotations

import copy

import numpy as np

from .schema import DatasetEpisode


def episode_to_minari_dict(episode: DatasetEpisode) -> dict:
    """Return Minari core arrays plus a lossless AIO-Gym sidecar."""

    observation = episode.array("observation")
    next_observation = episode.array("next_observation")
    observations = np.concatenate(
        (observation[:1], next_observation),
        axis=0,
    )
    return {
        "observations": observations,
        "actions": episode.array("action_policy_normalized").copy(),
        "rewards": episode.array("reward_scalar").copy(),
        "terminations": episode.array("terminated").copy(),
        "truncations": episode.array("truncated").copy(),
        "infos": [{} for _ in range(episode.transition_count + 1)],
        "aiogym_sidecar": {
            "episode": copy.deepcopy(episode.storage_metadata()),
            "arrays": {
                name: array.copy()
                for name, array in episode.arrays().items()
            },
        },
    }


def episode_from_minari_dict(payload) -> DatasetEpisode:
    """Restore a lossless sidecar and verify Minari episode boundaries."""

    sidecar = payload.get("aiogym_sidecar")
    if not isinstance(sidecar, dict):
        raise ValueError("Minari payload is missing aiogym_sidecar")
    episode = DatasetEpisode.from_storage(
        sidecar["episode"],
        sidecar["arrays"],
    )
    observations = np.asarray(payload["observations"])
    if observations.shape[0] != episode.transition_count + 1:
        raise ValueError("Minari observation boundary count is invalid")
    if not np.array_equal(
        observations[0],
        episode.array("observation")[0],
    ) or not np.array_equal(
        observations[1:],
        episode.array("next_observation"),
    ):
        raise ValueError("Minari observations do not match AIO-Gym sidecar")
    for payload_name, episode_name in (
        ("actions", "action_policy_normalized"),
        ("rewards", "reward_scalar"),
        ("terminations", "terminated"),
        ("truncations", "truncated"),
    ):
        if not np.array_equal(
            np.asarray(payload[payload_name]),
            episode.array(episode_name),
        ):
            raise ValueError(
                f"Minari {payload_name} do not match AIO-Gym sidecar"
            )
    return episode


def require_minari():
    """Import Minari only when an external export is explicitly requested."""

    try:
        import minari
    except ImportError as exc:
        raise ImportError(
            "Minari export requires the optional 'minari' package"
        ) from exc
    return minari


__all__ = [
    "episode_from_minari_dict",
    "episode_to_minari_dict",
    "require_minari",
]

"""Deterministic generalist sampling across resolved benchmark cases."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np

from .tracks import TrackSpec, load_track


SEED_COMPONENTS = (
    "selection",
    "initial",
    "reference",
    "disturbance",
    "noise",
    "plant",
)


def derive_seed_bundle(
    base_seed: int,
    namespace: str,
    *,
    worker_index: int = 0,
    episode_index: int = 0,
) -> dict[str, Any]:
    """Derive stable independent component seeds from one public base seed."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("seed namespace must be a non-empty string")
    base = int(base_seed)
    worker = int(worker_index)
    episode = int(episode_index)
    if worker < 0 or episode < 0:
        raise ValueError("worker_index and episode_index must be non-negative")

    prefix = f"{namespace}\0{base}\0{worker}\0{episode}".encode()
    namespace_hash = hashlib.sha256(namespace.encode()).hexdigest()
    bundle = {
        "base_seed": base,
        "namespace": namespace,
        "namespace_hash": namespace_hash,
        "worker_index": worker,
        "episode_index": episode,
    }
    for component in SEED_COMPONENTS:
        digest = hashlib.sha256(
            prefix + b"\0" + component.encode()
        ).digest()
        bundle[component] = int.from_bytes(digest[:8], "big") % (2**32)
    return bundle


class CaseMixtureEnv(gym.Env):
    """Sample one compatible resolved case at every episode reset."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        track: TrackSpec | str | Mapping[str, Any],
        *,
        split: str = "training",
        worker_index: int = 0,
        training: bool | None = None,
        env_factory=None,
    ) -> None:
        super().__init__()
        self.track = (
            track
            if isinstance(track, TrackSpec)
            else load_track(track)
        )
        self.split = str(split)
        self.worker_index = int(worker_index)
        if self.worker_index < 0:
            raise ValueError("worker_index must be non-negative")
        self.training = self.split == "training" if training is None else bool(
            training
        )
        if self.training and self.split != "training":
            raise ValueError(
                "training sampler cannot read validation or test seed namespace"
            )
        self.namespace = self.track.seed_namespace(self.split)
        self._cases = self.track.resolved_cases(self.split)
        if env_factory is None:
            from aiogym.env import AIOGymEnv

            env_factory = lambda case: AIOGymEnv(
                self.track.scenario,
                case=case.profile,
                reward_spec=self.track.reward_spec_id,
            )
        self._envs = tuple(env_factory(case) for case in self._cases)
        if not self._envs:
            raise ValueError("case mixture requires at least one resolved case")
        self._active_index = 0
        self._active_env = self._envs[0]
        self.action_space = self._active_env.action_space
        self.observation_space = self._active_env.observation_space
        self._base_seed: int | None = None
        self._episode_index = -1
        self._last_seed_bundle: dict[str, Any] | None = None
        self._validate_constructed_spaces()

    @property
    def active_case(self):
        return self._cases[self._active_index]

    @property
    def active_env(self):
        return self._active_env

    @property
    def last_seed_bundle(self):
        return deepcopy(self._last_seed_bundle)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._base_seed = int(seed)
            self._episode_index = 0
        else:
            if self._base_seed is None:
                self._base_seed = 0
                self._episode_index = 0
            else:
                self._episode_index += 1
        bundle = derive_seed_bundle(
            self._base_seed,
            self.namespace,
            worker_index=self.worker_index,
            episode_index=self._episode_index,
        )
        super().reset(seed=int(bundle["selection"]))
        weights = np.asarray(
            [case.weight for case in self._cases],
            dtype=np.float64,
        )
        probabilities = weights / np.sum(weights)
        self._active_index = int(
            self.np_random.choice(len(self._cases), p=probabilities)
        )
        self._active_env = self._envs[self._active_index]
        self._last_seed_bundle = bundle
        reset_options = dict(options or {})
        reset_options["seed_bundle"] = deepcopy(bundle)
        observation, info = self._active_env.reset(
            seed=int(bundle["initial"]),
            options=reset_options,
        )
        return observation, self._episode_info(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self._active_env.step(action)
        )
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._episode_info(info),
        )

    def close(self):
        for env in self._envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def render(self):
        return self._active_env.render()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        active = self.__dict__.get("_active_env")
        if active is None:
            raise AttributeError(name)
        return getattr(active, name)

    def _episode_info(self, info):
        return {
            **dict(info or {}),
            "track_id": self.track.id,
            "track_hash": self.track.track_hash,
            "track_split": self.split,
            "case_id": self.active_case.case_id,
            "resolved_case_hash": self.active_case.resolved_case_hash,
            "seed_bundle": deepcopy(self._last_seed_bundle),
        }

    def _validate_constructed_spaces(self):
        for case, env in zip(self._cases[1:], self._envs[1:]):
            if env.action_space != self.action_space:
                raise ValueError(
                    f"incompatible action space for case {case.case_id!r}"
                )
            if env.observation_space != self.observation_space:
                raise ValueError(
                    f"incompatible observation space for case {case.case_id!r}"
                )


__all__ = [
    "SEED_COMPONENTS",
    "CaseMixtureEnv",
    "derive_seed_bundle",
]

"""Sampler-driven training environment with coordinator-owned identities."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym

from aiogym._environment.builder import build_track_case_environment
from aiogym.generation.factory import make_episode_sampler
from aiogym.generation.samplers import FixedCaseEpisodeSampler

from .coordinator import EpisodeCoordinator


class _EpisodeSamplingEnv(gym.Wrapper):
    """Claim one global episode index and inject its resolved EpisodeSpec."""

    def __init__(
        self,
        env,
        *,
        sampler,
        coordinator: EpisodeCoordinator,
        worker_index: int = 0,
        track=None,
    ) -> None:
        super().__init__(env)
        self.sampler = sampler
        self.coordinator = coordinator
        self.worker_index = int(worker_index)
        if self.worker_index < 0:
            raise ValueError("worker_index must be non-negative")
        self.track = track
        self._assignment = None
        self._episode_spec = None
        self._public_reset_seed = None
        self._last_completed_episode_index = None

    @property
    def episode_assignment(self):
        return self._assignment

    @property
    def episode_spec(self):
        return self._episode_spec

    def reset(self, *, seed=None, options=None):
        if seed is not None and self._public_reset_seed is None:
            self._public_reset_seed = int(seed)
        reset_options = dict(options or {})
        if "episode_spec" in reset_options:
            raise ValueError(
                "_EpisodeSamplingEnv owns EpisodeSpec injection"
            )
        assignment = self.coordinator.claim(self.worker_index)
        episode = self.coordinator.sample_episode(
            self.sampler,
            assignment,
        )
        reset_options["episode_spec"] = episode
        observation, info = self.env.reset(
            seed=episode.base_seed,
            options=reset_options,
        )
        self._assignment = assignment
        self._episode_spec = episode
        return observation, self._episode_info(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        if (
            (terminated or truncated)
            and self._assignment is not None
        ):
            self._last_completed_episode_index = (
                self._assignment.episode_index
            )
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._episode_info(info),
        )

    def training_resume_state(self):
        """Return real coordinator progress for restart-episode checkpoints."""

        return {
            "coordinator": self.coordinator.state_dict(),
            "active_episode_index": (
                None
                if self._assignment is None
                else self._assignment.episode_index
            ),
            "last_completed_episode_index": (
                self._last_completed_episode_index
            ),
            "worker_index": self.worker_index,
        }

    def _episode_info(self, info):
        if self._assignment is None or self._episode_spec is None:
            return dict(info or {})
        payload = {
            **dict(info or {}),
            "episode_index": self._assignment.episode_index,
            "episode_worker_index": self.worker_index,
            "episode_seed_namespace": self._assignment.namespace,
            "episode_spec_id": self._episode_spec.episode_spec_id,
            "episode_spec_hash": self._episode_spec.resolved_hash,
            "distribution_id": self._episode_spec.distribution_id,
            "distribution_hash": self._episode_spec.distribution_hash,
        }
        if self.track is not None:
            payload.update(
                {
                    "track_id": self.track.id,
                    "track_hash": self.track.track_hash,
                    "track_split": "training",
                }
            )
        return payload


def make_track_episode_sampler(track):
    distribution_id = track.train_distribution_id
    if distribution_id is not None:
        return make_episode_sampler(
            track.training_distribution(),
            split="training",
        )
    if track.policy_scope == "generalist":
        raise ValueError(
            f"generalist Track {track.id!r} requires training.distribution_id"
        )
    cases = track.resolved_cases("training")
    if len(cases) != 1:
        raise ValueError("fixed-case training requires exactly one case")
    return FixedCaseEpisodeSampler(
        cases[0].profile,
        scenario=track.scenario,
        goal=track.goal,
        split="training",
    )


def make_track_training_env(
    track,
    *,
    base_seed: int,
    worker_index: int = 0,
    coordinator: EpisodeCoordinator | None = None,
    info_level: str = "minimal",
    profile_timing: bool = False,
):
    sampler = make_track_episode_sampler(track)
    resolved_coordinator = coordinator or EpisodeCoordinator(
        base_seed=base_seed,
        namespace=track.seed_namespace("training"),
    )
    base_env = make_track_training_base_env(
        track,
        sampler=sampler,
        info_level=info_level,
        profile_timing=profile_timing,
    )
    return _EpisodeSamplingEnv(
        base_env,
        sampler=sampler,
        coordinator=resolved_coordinator,
        worker_index=worker_index,
        track=track,
    )


def make_track_training_base_env(
    track,
    *,
    sampler=None,
    info_level: str = "minimal",
    profile_timing: bool = False,
):
    resolved_sampler = sampler or make_track_episode_sampler(track)
    profile = deepcopy(track.resolved_cases("training")[0].profile)
    profile["environment"]["control_dt"] = (
        resolved_sampler.distribution.control_dt
    )
    profile["environment"]["episode_steps"] = (
        resolved_sampler.distribution.episode_steps
    )
    return build_track_case_environment(
        track,
        SimpleNamespace(profile=profile),
        info_level=info_level,
        profile_timing=profile_timing,
    )


__all__ = [
    "_EpisodeSamplingEnv",
    "make_track_episode_sampler",
    "make_track_training_base_env",
    "make_track_training_env",
]

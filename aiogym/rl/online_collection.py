"""Batched online collection into the RLPD online replay."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .coordinator import EpisodeCoordinator
from .episode_env import _EpisodeSamplingEnv
from .replay import ReplayBuffer
from .statistics import NormalizedActionWrapper


class VectorOnlineCollector:
    """Synchronous vector collector with batched policy inference."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], object]],
        replay: ReplayBuffer,
        *,
        base_seed: int,
        namespace: str,
        sampler=None,
        track=None,
        coordinator_state=None,
    ) -> None:
        if not env_fns:
            raise ValueError("vector collector requires at least one env")
        restart_index = 0
        if coordinator_state is not None:
            state = dict(coordinator_state)
            if state.get("resume_mode") != "restart_episode":
                raise ValueError(
                    "vector collector supports only restart_episode resume"
                )
            if int(state["base_seed"]) != int(base_seed):
                raise ValueError("resume coordinator base_seed changed")
            if str(state["namespace"]) != str(namespace):
                raise ValueError("resume coordinator namespace changed")
            restart_index = int(state["next_episode_index"])
        coordinators = tuple(
            EpisodeCoordinator(
                base_seed=base_seed,
                namespace=namespace,
                next_episode_index=restart_index + worker,
                stride=len(env_fns),
            )
            for worker in range(len(env_fns))
        )
        envs = []
        for worker, (factory, coordinator) in enumerate(
            zip(env_fns, coordinators)
        ):
            env = factory()
            if sampler is not None:
                env = _EpisodeSamplingEnv(
                    env,
                    sampler=sampler,
                    coordinator=coordinator,
                    worker_index=worker,
                    track=track,
                )
            envs.append(NormalizedActionWrapper(env))
        self.envs = tuple(envs)
        self.replay = replay
        self.coordinators = coordinators
        self.coordinator = coordinators[0]
        self.sampler = sampler
        self.observations = []
        self.infos = []
        self.online_transitions = 0
        self.episodes = 0
        for worker, env in enumerate(self.envs):
            if self.sampler is None:
                assignment = self.coordinator.claim(worker)
                reset_seed = assignment.episode_seed
            else:
                reset_seed = base_seed
            observation, info = env.reset(seed=reset_seed)
            self.observations.append(np.asarray(observation, dtype=np.float32))
            self.infos.append(dict(info or {}))

    @property
    def n_envs(self) -> int:
        return len(self.envs)

    def collect(self, policy, transition_count: int) -> dict[str, int]:
        if transition_count < 0:
            raise ValueError("transition_count must be non-negative")
        remaining = int(transition_count)
        while remaining:
            active = min(self.n_envs, remaining)
            observation_batch = np.asarray(self.observations[:active])
            action_batch = _policy_actions(policy, observation_batch)
            if action_batch.shape[0] != active:
                raise ValueError("policy batch size does not match active envs")
            for worker in range(active):
                env = self.envs[worker]
                observation = self.observations[worker]
                action = np.asarray(action_batch[worker], dtype=np.float32)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
                self.replay.add(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                    bootstrap_mask=0.0 if terminated else 1.0,
                )
                self.online_transitions += 1
                remaining -= 1
                if terminated or truncated:
                    self.episodes += 1
                    if self.sampler is None:
                        assignment = self.coordinator.claim(worker)
                        next_observation, info = env.reset(
                            seed=assignment.episode_seed
                        )
                    else:
                        next_observation, info = env.reset()
                self.observations[worker] = np.asarray(
                    next_observation,
                    dtype=np.float32,
                )
                self.infos[worker] = dict(info or {})
        return self.accounting()

    def accounting(self) -> dict[str, int]:
        return {
            "n_envs": self.n_envs,
            "online_transitions": self.online_transitions,
            "completed_episodes": self.episodes,
            "online_replay_transitions": len(self.replay),
        }

    def resume_state(self) -> dict:
        """Capture real active coordinators for restart-episode resume."""

        worker_states = []
        for env in self.envs:
            if self.sampler is None:
                worker_states.append(
                    {
                        "coordinator": self.coordinator.state_dict(),
                        "active_episode_index": None,
                        "last_completed_episode_index": None,
                    }
                )
                continue
            worker_states.append(env.training_resume_state())
        active = [
            int(state["active_episode_index"])
            for state in worker_states
            if state.get("active_episode_index") is not None
        ]
        completed = [
            int(state["last_completed_episode_index"])
            for state in worker_states
            if state.get("last_completed_episode_index") is not None
        ]
        return {
            "resume_mode": "restart_episode",
            "base_seed": self.coordinators[0].base_seed,
            "namespace": self.coordinators[0].namespace,
            "worker_states": worker_states,
            "last_committed_episode_index": (
                max(completed) if completed else None
            ),
            "next_episode_index": (
                max(active) + 1
                if active
                else max(
                    coordinator.next_episode_index
                    for coordinator in self.coordinators
                )
            ),
            "active_episode_indexes": active,
            "partial_episodes_discarded": len(active),
            "n_envs": self.n_envs,
            "vector_backend": "synchronous_vector",
        }

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _policy_actions(policy, observations) -> np.ndarray:
    if hasattr(policy, "policy_action_batch"):
        actions = policy.policy_action_batch(observations)
    elif hasattr(policy, "policy_action"):
        actions = [policy.policy_action(value) for value in observations]
    elif callable(policy):
        actions = policy(observations)
    else:
        raise TypeError("policy must expose policy_action_batch or be callable")
    values = np.asarray(actions, dtype=np.float32)
    return np.clip(values, -1.0, 1.0)


__all__ = ["VectorOnlineCollector"]

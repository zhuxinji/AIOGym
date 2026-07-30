"""Coordinator-owned episode identities for deterministic vector rollout."""
from __future__ import annotations

from dataclasses import dataclass

from aiogym.generation import SeedTree


@dataclass(frozen=True)
class EpisodeAssignment:
    episode_index: int
    worker_index: int
    base_seed: int
    namespace: str
    episode_seed: int


class EpisodeCoordinator:
    """Assign monotonically indexed episodes; workers only consume work.

    Worker placement is deliberately absent from seed derivation. Therefore the
    episode identity sequence is stable when ``n_envs`` changes.
    """

    def __init__(
        self,
        *,
        base_seed: int,
        namespace: str,
        next_episode_index: int = 0,
        stride: int = 1,
    ) -> None:
        self.base_seed = int(base_seed)
        self.namespace = str(namespace)
        self.next_episode_index = int(next_episode_index)
        self.stride = int(stride)
        if self.base_seed < 0 or self.next_episode_index < 0:
            raise ValueError("coordinator seeds and indexes must be non-negative")
        if self.stride <= 0:
            raise ValueError("coordinator stride must be positive")
        if not self.namespace:
            raise ValueError("coordinator namespace must be non-empty")

    def claim(self, worker_index: int = 0) -> EpisodeAssignment:
        worker = int(worker_index)
        if worker < 0:
            raise ValueError("worker_index must be non-negative")
        episode_index = self.next_episode_index
        self.next_episode_index += self.stride
        tree = SeedTree(
            self.base_seed,
            self.namespace,
            worker_index=0,
            episode_index=episode_index,
        )
        return EpisodeAssignment(
            episode_index=episode_index,
            worker_index=worker,
            base_seed=self.base_seed,
            namespace=self.namespace,
            episode_seed=tree.seed("initial_state"),
        )

    def claim_many(self, count: int, *, n_envs: int) -> tuple[EpisodeAssignment, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        return tuple(self.claim(index % n_envs) for index in range(count))

    def sample_episode(self, sampler, assignment: EpisodeAssignment):
        """Resolve an assignment without including worker placement in identity."""

        return sampler.sample(
            assignment.base_seed,
            episode_index=assignment.episode_index,
        )

    def state_dict(self) -> dict[str, int | str]:
        return {
            "base_seed": self.base_seed,
            "namespace": self.namespace,
            "next_episode_index": self.next_episode_index,
            "stride": self.stride,
        }

    @classmethod
    def from_state_dict(cls, state) -> "EpisodeCoordinator":
        return cls(
            base_seed=int(state["base_seed"]),
            namespace=str(state["namespace"]),
            next_episode_index=int(state["next_episode_index"]),
            stride=int(state.get("stride", 1)),
        )


__all__ = ["EpisodeAssignment", "EpisodeCoordinator"]

"""Environment throughput profiling without optional system dependencies."""
from __future__ import annotations

import resource
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EnvironmentProfile:
    n_envs: int
    transitions: int
    elapsed_seconds: float
    reset_seconds: float
    step_seconds: float
    single_env_steps_per_second: float
    vector_transitions_per_second: float
    memory_megabytes: float
    memory_megabytes_per_worker: float
    ode_integration_seconds: float
    reward_metric_seconds: float
    info_construction_seconds: float
    observation_seconds: float
    dispatch_overhead_seconds: float
    cpu_utilization: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_envs": self.n_envs,
            "transitions": self.transitions,
            "elapsed_seconds": self.elapsed_seconds,
            "reset_seconds": self.reset_seconds,
            "step_seconds": self.step_seconds,
            "single_env_steps_per_second": self.single_env_steps_per_second,
            "vector_transitions_per_second": (
                self.vector_transitions_per_second
            ),
            "memory_megabytes": self.memory_megabytes,
            "memory_megabytes_per_worker": self.memory_megabytes_per_worker,
            "ode_integration_seconds": self.ode_integration_seconds,
            "reward_metric_seconds": self.reward_metric_seconds,
            "info_construction_seconds": self.info_construction_seconds,
            "observation_seconds": self.observation_seconds,
            "dispatch_overhead_seconds": self.dispatch_overhead_seconds,
            "cpu_utilization": self.cpu_utilization,
        }


def profile_environments(
    env_factory: Callable[[int], object],
    *,
    n_envs: int,
    transitions: int,
    seed: int = 0,
) -> EnvironmentProfile:
    """Profile synchronous vector work using the same transition semantics."""

    if n_envs <= 0 or transitions <= 0:
        raise ValueError("n_envs and transitions must be positive")
    envs = [env_factory(index) for index in range(n_envs)]
    reset_seconds = 0.0
    step_seconds = 0.0
    completed = 0
    observations = []
    component_totals = {
        "ode_integration_seconds": 0.0,
        "reward_metric_seconds": 0.0,
        "info_construction_seconds": 0.0,
        "observation_seconds": 0.0,
    }
    cpu_started = perf_counter()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    start = perf_counter()
    try:
        for index, env in enumerate(envs):
            before = perf_counter()
            observation, _ = env.reset(seed=seed + index)
            reset_seconds += perf_counter() - before
            observations.append(observation)
        while completed < transitions:
            for index, env in enumerate(envs):
                if completed >= transitions:
                    break
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                before = perf_counter()
                observation, _, terminated, truncated, _ = env.step(action)
                step_seconds += perf_counter() - before
                source = getattr(env, "active_env", env)
                timings = getattr(source, "last_step_timings", {})
                for name in component_totals:
                    component_totals[name] += float(timings.get(name, 0.0))
                completed += 1
                if terminated or truncated:
                    before = perf_counter()
                    observation, _ = env.reset()
                    reset_seconds += perf_counter() - before
                observations[index] = observation
    finally:
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    elapsed = perf_counter() - start
    usage_finished = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (
        usage_finished.ru_utime
        + usage_finished.ru_stime
        - usage_started.ru_utime
        - usage_started.ru_stime
    )
    cpu_wall_seconds = perf_counter() - cpu_started
    memory_mb = _resident_memory_megabytes()
    vector_rate = completed / elapsed if elapsed > 0.0 else float("inf")
    step_rate = completed / step_seconds if step_seconds > 0.0 else float("inf")
    measured_components = sum(component_totals.values())
    return EnvironmentProfile(
        n_envs=n_envs,
        transitions=completed,
        elapsed_seconds=elapsed,
        reset_seconds=reset_seconds,
        step_seconds=step_seconds,
        single_env_steps_per_second=step_rate,
        vector_transitions_per_second=vector_rate,
        memory_megabytes=memory_mb,
        memory_megabytes_per_worker=memory_mb / n_envs,
        **component_totals,
        dispatch_overhead_seconds=max(0.0, step_seconds - measured_components),
        cpu_utilization=(
            cpu_seconds / cpu_wall_seconds if cpu_wall_seconds > 0.0 else 0.0
        ),
    )


def _resident_memory_megabytes() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    if usage > 10_000_000:
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


__all__ = ["EnvironmentProfile", "profile_environments"]

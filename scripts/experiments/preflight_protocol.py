"""Run protocol checks without training or test-split rollouts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aiogym._internal.serialization import write_json_artifact
from aiogym.benchmarks.anchors import load_anchor_set
from aiogym.benchmarks.tracks.audit import require_split_isolation
from aiogym.evaluation.provenance import reward_spec_hash
from aiogym.experiments import load_training_config
from aiogym.rl.episode_env import make_track_training_env
from aiogym.rl.plan import resolve_training_plan


PREFLIGHT_SCHEMA_VERSION = "aiogym.protocol_preflight.v1"


def preflight_protocol(config_path: str | Path, *, samples: int = 3) -> dict:
    if samples < 2:
        raise ValueError("preflight samples must be at least two")
    plan = resolve_training_plan(load_training_config(config_path))
    track = plan.track
    require_split_isolation(track)
    anchor = load_anchor_set(
        track.ranking_declaration["anchor_id"],
        track=track,
    )
    actual_contract = track.validate_policy_contract()
    env = make_track_training_env(
        track,
        base_seed=plan.config.training_seed,
        info_level="full",
    )
    try:
        observation_space = env.observation_space
        action_space = env.action_space
        _require_finite_box("observation", observation_space)
        _require_finite_box("action", action_space)
        episode_hashes = []
        episode_samples = []
        observation = None
        reset_info = None
        for _ in range(samples):
            observation, reset_info = env.reset()
            episode_hashes.append(str(reset_info["episode_spec_hash"]))
            episode = env.episode_spec
            episode_samples.append(
                {
                    "episode_spec_hash": episode.resolved_hash,
                    "episode_steps": episode.episode_steps,
                    "reference_event_count": len(
                        episode.reference_schedule
                    ),
                    "difficulty_tags": list(episode.difficulty_tags),
                }
            )
        neutral = 0.5 * (
            np.asarray(action_space.low, dtype=np.float64)
            + np.asarray(action_space.high, dtype=np.float64)
        )
        stepped, reward, terminated, truncated, step_info = env.step(neutral)
        if not (
            np.all(np.isfinite(observation))
            and np.all(np.isfinite(stepped))
            and np.isfinite(reward)
        ):
            raise ValueError("preflight reset/step produced non-finite values")
        if len(set(episode_hashes)) < 2:
            raise ValueError("training EpisodeSpec sampling lacks diversity")
        if track.scenario == "cascade":
            contract = track.policy_contract
            if not contract["normalize_observations"]:
                raise ValueError("Cascade v2 observations must be normalized")
            if contract["disturbance_obs"]:
                raise ValueError(
                    "Cascade v2 disturbances must remain unmeasured"
                )
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "passed": True,
            "config_path": str(Path(config_path)),
            "config_hash": plan.config.config_hash,
            "resolved_config": plan.config.as_dict(),
            "track_id": track.id,
            "track_hash": track.track_hash,
            "reward_spec_id": track.reward_spec_id,
            "reward_spec_hash": reward_spec_hash(track.reward_spec_id),
            "anchor_id": anchor.anchor_id,
            "anchor_artifact_hash": anchor.artifact_hash,
            "validation_plan_hash": plan.validation_plan.plan_hash,
            "policy_contract": actual_contract,
            "observation_space": _space_metadata(observation_space),
            "action_space": _space_metadata(action_space),
            "training_episode_spec_hashes": episode_hashes,
            "training_distribution_id": env.sampler.distribution_id,
            "training_distribution_hash": env.sampler.distribution_hash,
            "training_episode_samples": episode_samples,
            "unique_training_episode_specs": len(set(episode_hashes)),
            "neutral_step": {
                "performed": True,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "episode_spec_hash": step_info.get("episode_spec_hash"),
            },
            "test_rollouts": 0,
            "optimizer_updates": 0,
        }
    finally:
        env.close()


def _require_finite_box(name, space) -> None:
    low = np.asarray(getattr(space, "low", ()), dtype=np.float64)
    high = np.asarray(getattr(space, "high", ()), dtype=np.float64)
    if low.size == 0 or low.shape != high.shape:
        raise ValueError(f"{name} space must expose matching Box bounds")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError(f"{name} space bounds must be finite")
    if np.any(high <= low):
        raise ValueError(f"{name} space bounds must be ordered")


def _space_metadata(space) -> dict:
    return {
        "shape": [int(size) for size in space.shape],
        "low": np.asarray(space.low).tolist(),
        "high": np.asarray(space.high).tolist(),
        "dtype": str(space.dtype),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    report = preflight_protocol(args.config, samples=args.samples)
    write_json_artifact(args.output, report)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

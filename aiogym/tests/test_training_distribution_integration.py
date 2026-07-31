from __future__ import annotations

from copy import deepcopy

import numpy as np

from aiogym.benchmarks import TrackSpec, load_track
from aiogym.generation import make_episode_sampler
from aiogym.rl.coordinator import EpisodeCoordinator
from aiogym.rl.episode_env import (
    _EpisodeSamplingEnv,
    make_track_episode_sampler,
    make_track_training_base_env,
    make_track_training_env,
)
from aiogym.rl.online_collection import VectorOnlineCollector
from aiogym.rl.replay import ReplayBuffer
from aiogym.rl.backends.rlpd import make_training_env as make_rlpd_env
from aiogym.rl.backends.sb3 import make_training_env as make_sb3_env
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.plan import resolve_training_plan


TRACK_ID = "quadruple-regulation-generalist-v1"


def _training_plan(n_envs=2):
    return resolve_training_plan(
        RLTrainingConfig(
            track_id=TRACK_ID,
            algorithm_id="sac",
            training_seed=5,
            total_transitions=1,
            n_envs=n_envs,
        )
    )


def test_track_distribution_changes_and_replays_episode_sequence():
    track = load_track(TRACK_ID)

    def sequence():
        env = make_track_training_env(track, base_seed=41)
        try:
            rows = []
            for reset_index in range(3):
                _, info = env.reset(seed=41 if reset_index == 0 else None)
                episode = env.episode_spec
                rows.append(
                    (
                        info["episode_index"],
                        info["episode_spec_hash"],
                        episode.plant_parameters,
                        episode.reference_schedule,
                        episode.disturbance_schedule,
                    )
                )
            return rows
        finally:
            env.close()

    first = sequence()
    second = sequence()
    assert first == second
    assert len({row[1] for row in first}) == len(first)
    assert any(
        first[index][2:] != first[0][2:]
        for index in range(1, len(first))
    )


def test_episode_identity_does_not_depend_on_worker_placement():
    sampler = make_episode_sampler(
        "quadruple-regulation-training-l2-v1"
    )
    first = sampler.sample(17, worker_index=0, episode_index=9)
    shifted = sampler.sample(17, worker_index=7, episode_index=9)
    assert first.episode_spec_id == shifted.episode_spec_id
    assert first.resolved_hash == shifted.resolved_hash


def test_first_episode_identity_set_is_invariant_to_env_count():
    track = load_track(TRACK_ID)
    sampler = make_track_episode_sampler(track)
    identity_sets = []
    for n_envs in (1, 2, 4, 8):
        coordinators = [
            EpisodeCoordinator(
                base_seed=12,
                namespace=track.seed_namespace("training"),
                next_episode_index=rank,
                stride=n_envs,
            )
            for rank in range(n_envs)
        ]
        identities = set()
        while len(identities) < 16:
            for coordinator in coordinators:
                if len(identities) == 16:
                    break
                assignment = coordinator.claim()
                identities.add(
                    coordinator.sample_episode(
                        sampler,
                        assignment,
                    ).resolved_hash
                )
        identity_sets.append(identities)
    assert all(rows == identity_sets[0] for rows in identity_sets[1:])


def test_sb3_and_rlpd_builders_use_episode_sampling_env():
    plan = _training_plan()
    track = plan.track
    sb3 = make_sb3_env(plan, rank=1)()
    rlpd = make_rlpd_env(plan, worker_index=1)
    try:
        assert _find_wrapper(sb3, _EpisodeSamplingEnv) is not None
        assert _find_wrapper(rlpd, _EpisodeSamplingEnv) is not None
        _, sb3_info = sb3.reset(seed=6)
        _, rlpd_info = rlpd.reset(seed=6)
        assert sb3_info["distribution_id"] == track.train_distribution_id
        assert rlpd_info["distribution_id"] == track.train_distribution_id
    finally:
        sb3.close()
        rlpd.close()


def test_online_collector_injects_sampler_episode_specs():
    track = load_track(TRACK_ID)
    sampler = make_track_episode_sampler(track)
    collector = VectorOnlineCollector(
        [
            lambda: make_track_training_base_env(
                track,
                sampler=sampler,
            )
            for _ in range(2)
        ],
        ReplayBuffer(8, seed=2),
        base_seed=5,
        namespace=track.seed_namespace("training"),
        sampler=sampler,
        track=track,
    )
    try:
        assert {info["episode_index"] for info in collector.infos} == {0, 1}
        assert {
            info["distribution_id"] for info in collector.infos
        } == {track.train_distribution_id}
        assert len(
            {info["episode_spec_hash"] for info in collector.infos}
        ) == 2
    finally:
        collector.close()


def test_sb3_vector_autoreset_advances_partitioned_global_indexes():
    from stable_baselines3.common.vec_env import DummyVecEnv

    plan = _training_plan()
    vector_env = DummyVecEnv(
        [make_sb3_env(plan, rank=rank) for rank in range(2)]
    )
    try:
        vector_env.reset()
        assert {
            info["episode_index"] for info in vector_env.reset_infos
        } == {0, 1}
        dones = np.zeros(2, dtype=bool)
        for _ in range(900):
            _, _, dones, _ = vector_env.step(
                np.zeros((2, 2), dtype=np.float32)
            )
        assert np.all(dones)
        assert {
            info["episode_index"] for info in vector_env.reset_infos
        } == {2, 3}
    finally:
        vector_env.close()


def test_generalist_track_cannot_fall_back_to_fixed_cases():
    declaration = deepcopy(
        load_track(
            TRACK_ID,
            validate_policy_contract=False,
        ).declaration
    )
    declaration["id"] = "generalist-without-distribution-v1"
    declaration["training"].pop("distribution_id")
    track = TrackSpec(declaration)
    try:
        make_track_episode_sampler(track)
    except ValueError as exc:
        assert "requires training.distribution_id" in str(exc)
    else:
        raise AssertionError("generalist Track silently used fixed cases")


def _find_wrapper(env, cls):
    current = env
    while current is not None:
        if isinstance(current, cls):
            return current
        current = getattr(current, "env", None)
    return None

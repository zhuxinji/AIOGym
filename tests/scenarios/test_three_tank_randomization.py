from __future__ import annotations

import numpy as np
import pytest

from aiogym import DatasetReader, collect, evaluate, make_env


def _randomized_env():
    return make_env(
        "three_tank",
        randomize=True,
        noise=True,
        delay=True,
        fault={"probability": 1.0},
    )


def test_randomization_is_seeded_and_records_resolved_episode_and_channels():
    first = _randomized_env()
    second = _randomized_env()
    different = _randomized_env()
    try:
        first_observation, first_info = first.reset(seed=41)
        second_observation, second_info = second.reset(seed=41)
        different_observation, different_info = different.reset(seed=42)
        assert np.array_equal(first_observation, second_observation)
        assert first_info["episode_spec"] == second_info["episode_spec"]
        assert first_info["runtime_variation"] == second_info["runtime_variation"]
        assert not np.array_equal(first_observation, different_observation)
        assert first_info["episode_spec"] != different_info["episode_spec"]
        assert first_info["episode_family"] == "tracking"
        variation = first_info["runtime_variation"]
        assert 0 <= variation["action_delay_steps"] <= 1
        assert 0 <= variation["observation_delay_steps"] <= 2
        assert variation["fault"]["kind"] == "loss-of-effectiveness"
    finally:
        first.close()
        second.close()
        different.close()


def test_episode_sampling_rng_is_independent_from_channel_options():
    plain = make_env("three_tank", randomize=True)
    varied = make_env("three_tank", randomize=True, noise=True, delay=True, fault=True)
    try:
        _, plain_info = plain.reset(seed=19)
        _, varied_info = varied.reset(seed=19)
    finally:
        plain.close()
        varied.close()
    assert plain_info["episode_family"] == varied_info["episode_family"]
    assert plain_info["episode_spec"] == varied_info["episode_spec"]


@pytest.mark.parametrize(
    ("scenario", "start_bounds", "duration_bounds", "factor_bounds"),
    (
        ("quadruple", (36, 72), (36, 72), (0.78, 0.94)),
        ("three_tank", (30, 90), (360, 360), None),
    ),
)
def test_physical_disturbance_is_seeded_and_changes_model_dynamics(
    scenario,
    start_bounds,
    duration_bounds,
    factor_bounds,
):
    clean = make_env(scenario)
    first = make_env(scenario, disturbance=True)
    repeated = make_env(scenario, disturbance=True)
    different = make_env(scenario, disturbance=True)
    try:
        clean_observation, clean_info = clean.reset(seed=23)
        first_observation, first_info = first.reset(seed=23)
        _, repeated_info = repeated.reset(seed=23)
        _, different_info = different.reset(seed=24)

        schedule = first_info["episode_spec"]["disturbance_schedule"]
        assert schedule == repeated_info["episode_spec"]["disturbance_schedule"]
        assert schedule != different_info["episode_spec"]["disturbance_schedule"]
        assert np.array_equal(first_observation, clean_observation)
        assert first_info["episode_family"] == "tracking"
        assert first.unwrapped.runtime_config["disturbance"] is True

        if scenario == "three_tank":
            assert all(
                set(values) <= {"bv12_open", "bv23_open", "bv34_open"}
                for values in schedule.values()
            )
            bypass_steps = sorted(
                int(step)
                for step, values in schedule.items()
                if {"bv12_open", "bv23_open", "bv34_open"} & set(values)
            )
            assert len(bypass_steps) == 2
            start, end = bypass_steps
            active_names = set(schedule[str(start)])
            assert 1 <= len(active_names) <= 3
            assert active_names <= {"bv12_open", "bv23_open", "bv34_open"}
            assert schedule[str(start)] == {name: 1.0 for name in active_names}
            assert schedule[str(end)] == {name: 0.0 for name in active_names}
            changed = {
                **clean.unwrapped.model.default_disturbances(),
                **{name: 1.0 for name in active_names},
            }
        else:
            pump_steps = sorted(
                int(step)
                for step, values in schedule.items()
                if "pump_flow_factor" in values
            )
            start, end = pump_steps
            factor = schedule[str(start)]["pump_flow_factor"]
            assert factor_bounds is not None
            assert factor_bounds[0] <= factor <= factor_bounds[1]
            assert schedule[str(end)]["pump_flow_factor"] == 1.0
            changed = {
                **clean.unwrapped.model.default_disturbances(),
                "pump_flow_factor": factor,
            }

        assert start_bounds[0] <= start <= start_bounds[1]
        assert duration_bounds[0] <= end - start <= duration_bounds[1]

        state = np.asarray(clean_info["episode_spec"]["initial_state"], dtype=float)
        action = np.asarray(clean_info["episode_spec"]["initial_action"], dtype=float)
        defaults = clean.unwrapped.model.default_disturbances()
        nominal_derivative = clean.unwrapped.model.dynamics(state, action, defaults)
        disturbed_derivative = clean.unwrapped.model.dynamics(state, action, changed)
        assert not np.allclose(disturbed_derivative, nominal_derivative)
    finally:
        clean.close()
        first.close()
        repeated.close()
        different.close()


def test_disturbance_rng_does_not_change_randomized_tracking_condition():
    plain = make_env("three_tank", randomize=True)
    disturbed = make_env("three_tank", randomize=True, disturbance=True)
    try:
        _, plain_info = plain.reset(seed=29)
        _, disturbed_info = disturbed.reset(seed=29)
    finally:
        plain.close()
        disturbed.close()
    disturbed_spec = dict(disturbed_info["episode_spec"])
    assert disturbed_spec["disturbance_schedule"]
    disturbed_spec["disturbance_schedule"] = {}
    assert disturbed_spec == plain_info["episode_spec"]


def test_three_tank_training_disturbance_samples_single_pair_and_triple_bypasses():
    env = make_env("three_tank", disturbance=True)
    combinations = set()
    try:
        for seed in range(200):
            _, info = env.reset(seed=seed)
            schedule = info["episode_spec"]["disturbance_schedule"]
            start = min(int(step) for step in schedule)
            combinations.add(tuple(sorted(schedule[str(start)])))
    finally:
        env.close()

    assert {len(names) for names in combinations} == {1, 2, 3}
    assert {names for names in combinations if len(names) == 1} == {
        ("bv12_open",),
        ("bv23_open",),
        ("bv34_open",),
    }
    assert {names for names in combinations if len(names) == 2} == {
        ("bv12_open", "bv23_open"),
        ("bv12_open", "bv34_open"),
        ("bv23_open", "bv34_open"),
    }
    assert {names for names in combinations if len(names) == 3} == {
        ("bv12_open", "bv23_open", "bv34_open"),
    }


def test_randomize_defaults_to_interior_tracking_only():
    env = make_env("three_tank", randomize=True)
    try:
        reset_infos = [env.reset(seed=seed)[1] for seed in range(200)]
    finally:
        env.close()
    assert {info["episode_family"] for info in reset_infos} == {"tracking"}
    assert {
        info["episode_parameters"]["initial_family"] for info in reset_infos
    } == {"interior"}
    assert all(not info["episode_spec"]["disturbance_schedule"] for info in reset_infos)
    moves = []
    for info in reset_infos:
        spec = info["episode_spec"]
        assert spec["horizon"] == 600
        assert not spec["reference_schedule"]
        start = np.asarray(spec["initial_state"], dtype=float)
        target = spec["reference"]
        level_move = np.abs(np.asarray(target, dtype=float) - np.asarray(start))
        moves.extend(level_move.tolist())
        assert np.all(level_move >= 0.05)
    assert max(moves) > 0.12


@pytest.mark.parametrize("scenario", ("three_tank", "quadruple"))
def test_boundary_probability_controls_the_training_tail(scenario):
    env = make_env(scenario, randomize=True, boundary_probability=0.30)
    try:
        reset_infos = [env.reset(seed=seed)[1] for seed in range(200)]
    finally:
        env.close()
    assert {info["episode_family"] for info in reset_infos} == {"tracking"}
    assert all(not info["episode_spec"]["disturbance_schedule"] for info in reset_infos)
    initial_families = [
        info["episode_parameters"]["initial_family"] for info in reset_infos
    ]
    assert initial_families.count("boundary-prerun") == 58
    assert initial_families.count("interior") == 142


def test_boundary_selection_rng_does_not_change_tracking_targets():
    interior = make_env("three_tank", randomize=True)
    boundary = make_env(
        "three_tank", randomize=True, boundary_probability=1.0
    )
    try:
        for seed in range(20):
            _, interior_info = interior.reset(seed=seed)
            _, boundary_info = boundary.reset(seed=seed)
            assert (
                interior_info["episode_spec"]["reference"]
                == boundary_info["episode_spec"]["reference"]
            )
            assert interior_info["episode_parameters"]["initial_family"] == "interior"
            assert (
                boundary_info["episode_parameters"]["initial_family"]
                == "boundary-prerun"
            )
    finally:
        interior.close()
        boundary.close()


def test_base_environment_does_not_randomize_automatically():
    env = make_env("three_tank")
    try:
        first_observation, first_info = env.reset(seed=1)
        second_observation, second_info = env.reset(seed=2)
    finally:
        env.close()
    assert np.array_equal(first_observation, second_observation)
    assert first_info["episode_spec"] == second_info["episode_spec"]
    assert first_info["episode_family"] == second_info["episode_family"] == "default"


def test_sampled_tracking_target_is_active_at_reset():
    env = make_env("three_tank", randomize=True)
    try:
        for seed in range(100):
            _, reset_info = env.reset(seed=seed)
            initial_state = np.asarray(
                reset_info["episode_spec"]["initial_state"], dtype=float
            )
            maximum = np.asarray(
                env.unwrapped.model.parameter("height_max"), dtype=float
            )
            if not np.all(initial_state >= 0.80 * maximum):
                break
        else:
            raise AssertionError("no interior tracking episode sampled")
        spec = reset_info["episode_spec"]
        target = spec["reference"]
        initial = spec["initial_state"]
        initial_action = np.asarray(
            spec["initial_action"], dtype=float
        )
        assert spec["horizon"] == 600
        assert not spec["reference_schedule"]
        assert np.all(0.125 <= np.asarray(initial))
        assert np.all(np.asarray(initial) <= 0.4)
        assert np.all(0.125 <= np.asarray(target))
        assert np.all(np.asarray(target) <= 0.4)
        assert np.all(np.abs(np.asarray(target) - np.asarray(initial)) >= 0.05)
        assert np.all((0.02 <= initial_action) & (initial_action <= 0.85))
        flow_lpm = (
            env.unwrapped.model.step_info(
                reset_info["episode_spec"]["initial_state"],
                initial_action,
                env.unwrapped.model.default_disturbances(),
            )["P101_flow_m3s"]
            * 60000.0
        )
        assert 1.0 <= flow_lpm <= 6.0
        assert np.allclose(env.unwrapped.y_sp, target)
    finally:
        env.close()


def test_randomized_direct_action_dataset_keeps_episode_metadata(tmp_path):
    env = make_env("three_tank", randomize=True)
    try:
        result = collect(
            env=env,
            policy="random",
            episodes=1,
            seed=17,
            max_steps=2,
            output=tmp_path / "tank3-dataset",
        )
    finally:
        env.close()
    episode = DatasetReader(result["path"])[0]
    assert episode.metadata["episode_family"] == "tracking"
    assert episode.metadata["episode_parameters"]["initial_family"] == "interior"
    assert episode.metadata["episode_spec"]["horizon"] > 0
    assert episode.array("action").shape == (2, 4)
    assert episode.array("channel_action").shape == (2, 4)


def test_randomized_direct_action_environment_uses_the_standard_evaluator():
    env = make_env("three_tank", randomize=True)
    try:
        result = evaluate(
            env=env,
            policy="random",
            seeds=[3, 4],
            max_steps=2,
        )
    finally:
        env.close()
    assert result["seeds"] == [3, 4]
    assert [episode["length"] for episode in result["episodes"]] == [2, 2]
    assert "tracking_ise" in result["aggregate"]

from __future__ import annotations

import pytest

from aiogym.rl.profiler import _resident_memory_megabytes, _rss_to_megabytes


def test_rss_unit_conversion_is_platform_explicit():
    assert _rss_to_megabytes(2 * 1024 * 1024, "darwin") == 2.0
    assert _rss_to_megabytes(2 * 1024, "linux") == 2.0
    assert _rss_to_megabytes(2 * 1024, "win32") is None


def test_windows_memory_is_explicitly_unavailable():
    assert _resident_memory_megabytes("win32") is None


@pytest.mark.parametrize("algorithm", ("sac", "td3", "ppo"))
def test_stable_subprocess_default_is_spawn(algorithm):
    from aiogym.rl.config import RLTrainingConfig, resolve_training_defaults

    config = RLTrainingConfig(
        track_id="portability",
        algorithm_id=algorithm,
        training_seed=0,
        total_transitions=10,
        n_envs=2,
    )
    assert resolve_training_defaults(config).algorithm["subproc_start_method"] == "spawn"


def test_invalid_subprocess_start_method_is_rejected():
    from aiogym.rl.config import RLTrainingConfig

    with pytest.raises(ValueError, match="subproc_start_method"):
        RLTrainingConfig(
            track_id="portability",
            algorithm_id="sac",
            training_seed=0,
            total_transitions=10,
            n_envs=2,
            algorithm={"subproc_start_method": "unsafe"},
        )

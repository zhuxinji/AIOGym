from __future__ import annotations

from importlib.resources import files

import pytest

from aiogym._internal.config import parse_count
from aiogym.datasets.profiles import (
    build_collection_config,
    list_collection_profiles,
)


@pytest.mark.parametrize("profile", list_collection_profiles())
def test_quick_collection_profile_comes_from_packaged_resource(profile):
    prefix = "package:aiogym/"
    assert profile.source.startswith(prefix)
    resource = files("aiogym").joinpath(profile.source.removeprefix(prefix))
    assert resource.is_file()
    assert profile.default_transitions > 0
    assert profile.default_workers > 0


def test_collection_profile_builds_canonical_config():
    config, profile = build_collection_config(
        "quadruple",
        "quick",
        base_seed=4,
        transitions=10_000,
        workers=2,
        output=None,
        dataset_id=None,
    )
    assert config.track_id == "quadruple-regulation-generalist-v1"
    assert config.target_transitions == 10_000
    assert config.workers == 2
    assert config.dataset_id == "quadruple-quick-seed4"
    assert str(config.output) == "datasets/quadruple-quick-seed4"
    assert profile.id == "quick-v1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("10000", 10_000), ("10k", 10_000), ("1m", 1_000_000), ("1.5m", 1_500_000)),
)
def test_parse_count_accepts_exact_suffixes(raw, expected):
    assert parse_count(raw, option="--transitions") == expected


@pytest.mark.parametrize("raw", ("0", "-1", "True", "nan", "1g", "1.5"))
def test_parse_count_rejects_invalid_values(raw):
    with pytest.raises((TypeError, ValueError)):
        parse_count(raw, option="--transitions")


def test_collection_profile_does_not_guess_recovery_distribution():
    with pytest.raises(ValueError, match="available combinations"):
        build_collection_config(
            "cascade:recovery",
            "quick",
            base_seed=0,
            transitions=None,
            workers=None,
            output=None,
            dataset_id=None,
        )

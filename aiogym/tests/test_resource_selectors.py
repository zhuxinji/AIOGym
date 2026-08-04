from __future__ import annotations

from pathlib import Path

import pytest

from aiogym.benchmarks.tracks.registry import (
    TRACK_SELECTORS,
    _validate_track_selectors,
    load_track,
    resolve_track_selector,
)


@pytest.mark.parametrize("selector,track_id", TRACK_SELECTORS.items())
def test_track_selector_resolves_to_canonical_identity(selector, track_id):
    short = load_track(selector)
    exact = load_track(track_id)
    assert resolve_track_selector(selector) == track_id
    assert short.id == track_id
    assert short.track_hash == exact.track_hash
    assert short.declaration == exact.declaration


def test_track_selector_registry_rejects_duplicate_target():
    with pytest.raises(ValueError, match="only one selector"):
        _validate_track_selectors(
            {
                "first": "quadruple-regulation-generalist-v1",
                "second": "quadruple-regulation-generalist-v1",
            }
        )


def test_track_selector_registry_rejects_missing_track():
    with pytest.raises(ValueError, match="unknown Track"):
        _validate_track_selectors({"missing": "not-a-track"})


def test_track_custom_path_still_loads(tmp_path):
    canonical = load_track("quadruple-regulation-generalist-v1")
    path = tmp_path / "custom.json"
    import json

    path.write_text(json.dumps(canonical.declaration), encoding="utf-8")
    loaded = load_track(Path(path))
    assert loaded.declaration == canonical.declaration


def test_unknown_track_selector_lists_discovery_options():
    with pytest.raises(FileNotFoundError, match="list tracks --ids") as exc:
        load_track("not-a-track")
    assert "quadruple" in str(exc.value)

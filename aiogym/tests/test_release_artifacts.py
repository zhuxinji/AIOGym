from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile
import tarfile


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("check_release_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def _archive(tmp_path, *extra: str, missing: str | None = None) -> Path:
    path = tmp_path / "aiogym-0.1.0-py3-none-any.whl"
    names = [name for name in CHECKER.REQUIRED_PATHS if name != missing]
    with zipfile.ZipFile(path, "w") as archive:
        for name in (*names, *extra):
            archive.writestr(name, "fixture")
    return path


def test_release_checker_accepts_complete_clean_archive(tmp_path):
    assert CHECKER.check_archive(_archive(tmp_path)) == []


def test_release_checker_reports_missing_required_file(tmp_path):
    errors = CHECKER.check_archive(
        _archive(tmp_path, missing="aiogym/controllers/export.py")
    )
    assert errors == ["missing required path: aiogym/controllers/export.py"]


def test_release_checker_rejects_generated_content(tmp_path):
    errors = CHECKER.check_archive(
        _archive(tmp_path, "aiogym/__pycache__/factory.pyc")
    )
    assert any("forbidden path" in error for error in errors)


def test_release_checker_rejects_legacy_runtime_module(tmp_path):
    errors = CHECKER.check_archive(
        _archive(tmp_path, "aiogym/_environment/runtime.py")
    )
    assert errors == [
        "legacy module is forbidden: aiogym/_environment/runtime.py"
    ]


def test_release_checker_allows_only_setuptools_sdist_source_manifest(tmp_path):
    path = tmp_path / "aiogym-0.1.0.tar.gz"
    staging = tmp_path / "staging"
    for name in (*CHECKER.REQUIRED_PATHS, "aiogym.egg-info/SOURCES.txt"):
        target = staging / "aiogym-0.1.0" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture", encoding="utf-8")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(staging / "aiogym-0.1.0", arcname="aiogym-0.1.0")
    assert CHECKER.check_archive(path) == []

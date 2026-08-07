"""Setuptools hooks used only while building release artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.sdist import sdist as _sdist


class CleanSdist(_sdist):
    """Keep generated egg-info out of the published source archive."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        shutil.rmtree(Path(base_dir) / "aiogym.egg-info", ignore_errors=True)


setup(cmdclass={"sdist": CleanSdist})

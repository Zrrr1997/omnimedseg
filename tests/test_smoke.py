"""Fast checks that need no data: imports, CLI, and registry consistency."""

import importlib
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import src
from src.core.constants import DOWNLOAD_URLS, MODALITY_MAP

# Modality key in constants.py -> converter subpackage
MODALITY_PACKAGES = {
    "Fundus": "fundus",
    "Ultrasound": "ultrasound",
    "Endoscopy": "endoscopy",
    "OCT": "oct",
    "Microscopy": "microscopy",
    "Dermoscopy": "dermoscopy",
    "XRay": "xray",
    "CT": "ct",
    "MRI": "mri",
}


def _all_modules():
    return sorted(m.name for m in pkgutil.walk_packages(src.__path__, "src."))


@pytest.mark.parametrize("module", _all_modules())
def test_module_imports(module):
    importlib.import_module(module)


def test_cli_help():
    exe = shutil.which("omnimedseg", path=str(Path(sys.executable).parent))
    assert exe, "console script 'omnimedseg' not installed in this environment"
    result = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_modalities_match_converter_packages():
    assert set(DOWNLOAD_URLS) == set(MODALITY_PACKAGES)


def _converter_registry(modality):
    return importlib.import_module(f"src.converters.{MODALITY_PACKAGES[modality]}").CONVERTERS


@pytest.mark.parametrize("modality", sorted(MODALITY_PACKAGES))
def test_every_dataset_has_a_converter(modality):
    registry = _converter_registry(modality)
    missing = sorted(name for name in DOWNLOAD_URLS[modality] if name not in registry)
    assert not missing, f"{modality}: datasets with a URL but no converter: {missing}"


@pytest.mark.xfail(
    strict=True,
    reason="Known issue: auto-discovery registers helper functions named convert_* "
    "(e.g. convert_rtstruct, convert_msd) as datasets. Fixed in the dataset-registry step.",
)
def test_no_converters_without_dataset():
    stray = {}
    for modality in MODALITY_PACKAGES:
        extra = sorted(set(_converter_registry(modality)) - set(DOWNLOAD_URLS[modality]))
        if extra:
            stray[modality] = extra
    assert not stray, f"converters that are not datasets: {stray}"


def test_dataset_names_unique_across_modalities():
    total = sum(len(urls) for urls in DOWNLOAD_URLS.values())
    assert total == len(MODALITY_MAP), "a dataset name appears in more than one modality"

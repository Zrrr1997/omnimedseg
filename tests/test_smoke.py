"""Fast checks that need no data: imports, CLI, and registry consistency."""

import importlib
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import omnimedseg
from omnimedseg.core.cards import MODALITIES, load_cards

MODALITY_MAP = {name: card.modality for name, card in load_cards().items()}

from omnimedseg.converters.registry import MODALITY_PACKAGES, _defined_converters, get_converter


def _all_modules():
    return sorted(m.name for m in pkgutil.walk_packages(omnimedseg.__path__, "omnimedseg."))


@pytest.mark.parametrize("module", _all_modules())
def test_module_imports(module):
    importlib.import_module(module)


def test_cli_help():
    exe = shutil.which("omnimedseg", path=str(Path(sys.executable).parent))
    assert exe, "console script 'omnimedseg' not installed in this environment"
    result = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_modalities_match_converter_packages():
    assert set(MODALITIES) == set(MODALITY_PACKAGES)


@pytest.mark.parametrize("dataset", sorted(MODALITY_MAP))
def test_every_dataset_resolves_to_exactly_one_converter(dataset):
    converter = get_converter(MODALITY_MAP[dataset], dataset)
    assert callable(converter)


def test_dataset_converters_take_no_required_arguments():
    """The pipeline calls converter() without arguments."""
    import inspect

    bad = []
    for dataset, modality in MODALITY_MAP.items():
        sig = inspect.signature(get_converter(modality, dataset))
        if any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in sig.parameters.values()):
            bad.append(dataset)
    assert not bad, f"converters with required arguments: {bad}"


def test_helpers_are_not_mistaken_for_datasets():
    with pytest.raises(LookupError):
        get_converter("CT", "RTSTRUCTS")
    with pytest.raises(LookupError):
        get_converter("CT", "DICOM_SEGS")

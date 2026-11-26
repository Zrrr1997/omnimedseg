"""Find and run the converter function for a dataset.

A dataset ``FOO`` of modality ``CT`` is converted by the function ``convert_foo`` that is
*defined* (not merely imported) in one of the modules of ``omnimedseg.converters.ct``.
Helper functions such as ``convert_rtstruct`` are never picked up because lookups go by
dataset name, and a dataset defined in two modules raises instead of resolving by chance.
"""

import importlib
import inspect
import logging
import os
import pkgutil
from functools import lru_cache
from typing import Callable, Dict, List

from omnimedseg.core.errors import ConverterNotFoundError

logger = logging.getLogger(__name__)

# Modality (as used in dataset cards) -> converter subpackage
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

# 3D datasets whose image/label pairs are re-aligned after conversion (CT, MRI), minus exceptions.
_ALIGN_MODALITIES = {"CT", "MRI"}
_SKIP_ALIGN = {"ADRENAL_ACC_KIT67", "CC_TUMOR_HETEROGENEITY"}


@lru_cache(maxsize=None)
def _defined_converters(modality: str) -> Dict[str, List[Callable]]:
    """Map DATASET_NAME -> list of convert_* functions defined in the modality's modules."""
    if modality not in MODALITY_PACKAGES:
        raise ConverterNotFoundError(f"Unknown modality '{modality}'")
    package = importlib.import_module(f"omnimedseg.converters.{MODALITY_PACKAGES[modality]}")
    found: Dict[str, List[Callable]] = {}
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("convert_") and func.__module__ == module.__name__:
                found.setdefault(name[len("convert_"):].upper(), []).append(func)
    return found


def get_converter(modality: str, dataset: str) -> Callable:
    """Return the converter for ``dataset``; raise if there is none or more than one."""
    candidates = _defined_converters(modality).get(dataset.upper(), [])
    if not candidates:
        raise ConverterNotFoundError(f"No converter for dataset '{dataset}' (modality '{modality}')")
    if len(candidates) > 1:
        modules = sorted(f.__module__ for f in candidates)
        raise ConverterNotFoundError(f"Dataset '{dataset}' has converters in several modules: {modules}")
    return candidates[0]


def convert_dataset(modality: str, dataset: str) -> None:
    """Run the converter for ``dataset`` plus any modality-specific post-processing."""
    name = dataset.upper()
    converter = get_converter(modality, name)
    logger.info(f"🚀 Converting {name} with {converter.__module__}.{converter.__name__}")
    converter()
    if modality in _ALIGN_MODALITIES and name not in _SKIP_ALIGN:
        from omnimedseg.converters.utils_ct import align_all_pairs

        align_all_pairs(os.path.join("datasets", modality, name, "converted"))

"""Where OmniMedSeg keeps downloaded, extracted and converted data.

The data root is resolved at call time, in this order:
  1. ``set_data_root(path)`` (the CLI calls this for ``--data-root``)
  2. the ``OMNIMEDSEG_DATA_ROOT`` environment variable
  3. ``datasets`` (relative to the current working directory; the historical default)

Each dataset lives in ``<data_root>/<Modality>/<NAME>/`` with ``compressed``/``extracted``/``converted``
sub-folders.
"""

import os
from typing import Optional

ENV_VAR = "OMNIMEDSEG_DATA_ROOT"
DEFAULT_DATA_ROOT = "datasets"

_data_root: Optional[str] = None


def set_data_root(path: Optional[str]) -> None:
    """Override the data root for this process (``None`` resets to env var / default).

    Also exported as ``OMNIMEDSEG_DATA_ROOT`` so helper scripts run in subprocesses use the same root.
    """
    global _data_root
    _data_root = path
    if path is None:
        os.environ.pop(ENV_VAR, None)
    else:
        os.environ[ENV_VAR] = path


def data_root() -> str:
    return _data_root or os.environ.get(ENV_VAR) or DEFAULT_DATA_ROOT


def dataset_dir(modality: str, name: str) -> str:
    return os.path.join(data_root(), modality, name)

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))


@pytest.fixture(autouse=True)
def _default_data_root(monkeypatch):
    """Every test starts from the default data root, regardless of the caller's environment."""
    from omnimedseg.core import paths

    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "_data_root", None)
    yield
    monkeypatch.setattr(paths, "_data_root", None)

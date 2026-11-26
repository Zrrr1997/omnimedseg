"""Data root resolution."""

import os

from omnimedseg.core import paths


def test_default_is_relative_datasets():
    assert paths.data_root() == "datasets"
    assert paths.dataset_dir("CT", "KITS23") == os.path.join("datasets", "CT", "KITS23")


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, "/data/omnimedseg")
    assert paths.data_root() == "/data/omnimedseg"


def test_set_data_root_overrides_env_and_exports_it(monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, "/from/env")
    paths.set_data_root("/from/cli")
    assert paths.data_root() == "/from/cli"
    assert os.environ[paths.ENV_VAR] == "/from/cli"  # visible to helper subprocesses (KiTS downloaders)
    paths.set_data_root(None)
    assert paths.data_root() == "datasets"


def test_pipeline_uses_data_root(tmp_path, monkeypatch):
    """Converted output of a (fake) dataset lands under the configured root, not the CWD."""
    from omnimedseg.core import pipeline

    root = tmp_path / "root"
    raw = root / "Fundus" / "HRF" / "extracted"
    raw.mkdir(parents=True)
    (raw / "x.png").write_bytes(b"x")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    paths.set_data_root(str(root))
    monkeypatch.setattr(pipeline, "convert_dataset",
                        lambda m, d: os.makedirs(os.path.join(paths.data_root(), m, d, "converted")))
    pipeline.process_datasets(["HRF"], "Fundus", convert=True)
    assert (root / "Fundus" / "HRF" / "converted").is_dir()
    assert not raw.exists()  # raw data removed under the root after success
    assert not (cwd / "datasets").exists()

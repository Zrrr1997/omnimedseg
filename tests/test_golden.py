"""Re-convert small real datasets and compare against checksums recorded from the original code.

Needs the raw data on disk (download once with `--download --convert --keep_raw_data`).
Datasets whose raw data is missing are skipped. Run with:  pytest -m golden
"""

import json
import shutil

import pytest

from golden_utils import GOLDEN_DATASETS, REPO_ROOT, golden_path, hash_converted

IDS = [d for _, d in GOLDEN_DATASETS]


def _convert_and_compare(modality, dataset, ds_dir, root_prefix=None):
    from omnimedseg.core.pipeline import process_datasets

    shutil.rmtree(ds_dir / "converted", ignore_errors=True)
    process_datasets([dataset], modality, convert=True, keep_raw_data=True)

    expected = json.loads(golden_path(dataset).read_text())
    actual = hash_converted(ds_dir / "converted", root_prefix)

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
    assert not (missing or extra or changed), (
        f"{dataset}: {len(missing)} missing, {len(extra)} extra, {len(changed)} changed files. "
        f"First few — missing: {missing[:5]}, extra: {extra[:5]}, changed: {changed[:5]}"
    )


def _require_raw(modality, dataset):
    ds_dir = REPO_ROOT / "datasets" / modality / dataset
    if not (ds_dir / "extracted").is_dir():
        pytest.skip(f"raw data for {dataset} not found at {ds_dir / 'extracted'}")
    if not golden_path(dataset).exists():
        pytest.skip(f"no golden checksums for {dataset}; run tests/make_golden.py")
    return ds_dir


@pytest.mark.golden
@pytest.mark.parametrize("modality,dataset", GOLDEN_DATASETS, ids=IDS)
def test_conversion_matches_golden(modality, dataset, monkeypatch):
    """Default setup: run from the repo root with the relative ./datasets root."""
    ds_dir = _require_raw(modality, dataset)
    monkeypatch.chdir(REPO_ROOT)
    _convert_and_compare(modality, dataset, ds_dir)


@pytest.mark.golden
@pytest.mark.parametrize("modality,dataset", GOLDEN_DATASETS, ids=IDS)
def test_conversion_with_custom_data_root(modality, dataset, monkeypatch, tmp_path):
    """Run from an unrelated directory with an absolute data root: output must be identical."""
    ds_dir = _require_raw(modality, dataset)
    root = (REPO_ROOT / "datasets").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMNIMEDSEG_DATA_ROOT", str(root))
    _convert_and_compare(modality, dataset, root / modality / dataset, root_prefix=str(root))
    assert not (tmp_path / "datasets").exists(), "nothing may be written relative to the working directory"

"""Re-convert small real datasets and compare against checksums recorded from the original code.

Needs the raw data on disk (download once with `--download --convert --keep_raw_data`).
Datasets whose raw data is missing are skipped. Run with:  pytest -m golden
"""

import json
import os
import shutil

import pytest

from golden_utils import GOLDEN_DATASETS, REPO_ROOT, golden_path, hash_converted


@pytest.mark.golden
@pytest.mark.parametrize("modality,dataset", GOLDEN_DATASETS, ids=[d for _, d in GOLDEN_DATASETS])
def test_conversion_matches_golden(modality, dataset, monkeypatch):
    ds_dir = REPO_ROOT / "datasets" / modality / dataset
    if not (ds_dir / "extracted").is_dir():
        pytest.skip(f"raw data for {dataset} not found at {ds_dir / 'extracted'}")
    if not golden_path(dataset).exists():
        pytest.skip(f"no golden checksums for {dataset}; run tests/make_golden.py")

    # Converters currently resolve 'datasets/...' relative to the working directory.
    monkeypatch.chdir(REPO_ROOT)
    from omnimedseg.core.pipeline import process_datasets

    shutil.rmtree(ds_dir / "converted", ignore_errors=True)
    process_datasets([dataset], modality, convert=True, keep_raw_data=True)

    expected = json.loads(golden_path(dataset).read_text())
    actual = hash_converted(ds_dir / "converted")

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
    assert not (missing or extra or changed), (
        f"{dataset}: {len(missing)} missing, {len(extra)} extra, {len(changed)} changed files. "
        f"First few — missing: {missing[:5]}, extra: {extra[:5]}, changed: {changed[:5]}"
    )

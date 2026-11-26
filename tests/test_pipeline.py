"""Pipeline safety: failures are reported, raw data is never lost, partial output never looks finished."""

import os

import pytest

from omnimedseg.core import pipeline
from omnimedseg.core.errors import DownloadError, ManualDownloadRequired, PipelineError


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run the pipeline inside an empty temporary directory (it resolves 'datasets/' from the CWD)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_raw(workdir, modality="Fundus", name="HRF"):
    extracted = workdir / "datasets" / modality / name / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "image.png").write_bytes(b"not-empty")
    return extracted


def test_failing_converter_keeps_raw_data_and_moves_partial_output(workdir, monkeypatch):
    extracted = _make_raw(workdir)
    converted = workdir / "datasets" / "Fundus" / "HRF" / "converted"

    def broken_converter(modality, dataset):
        (converted / "images").mkdir(parents=True)
        (converted / "images" / "000000.png").write_bytes(b"half")
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "convert_dataset", broken_converter)
    with pytest.raises(PipelineError) as exc:
        pipeline.process_datasets(["HRF"], "Fundus", convert=True)

    assert isinstance(exc.value.failures["HRF"], RuntimeError)
    assert extracted.is_dir(), "raw data must survive a failed conversion"
    assert not converted.exists(), "partial output must not look like a finished conversion"
    assert (converted.parent / "converted.failed" / "images" / "000000.png").exists()


def test_missing_converter_keeps_raw_data(workdir):
    extracted = _make_raw(workdir, name="NOT_A_DATASET")
    with pytest.raises(PipelineError) as exc:
        pipeline.process_datasets(["NOT_A_DATASET"], "Fundus", convert=True)
    assert isinstance(exc.value.failures["NOT_A_DATASET"], LookupError)
    assert extracted.is_dir()


def test_successful_conversion_removes_raw_and_stale_failed_output(workdir, monkeypatch):
    extracted = _make_raw(workdir)
    ds = workdir / "datasets" / "Fundus" / "HRF"
    (ds / "converted.failed").mkdir()
    monkeypatch.setattr(pipeline, "convert_dataset", lambda m, d: (ds / "converted").mkdir())

    pipeline.process_datasets(["HRF"], "Fundus", convert=True)
    assert (ds / "converted").is_dir()
    assert not (ds / "converted.failed").exists()
    assert not extracted.exists()


def test_one_failure_does_not_stop_other_datasets(workdir, monkeypatch):
    done = []

    def converter(modality, dataset):
        if dataset == "HRF":
            raise RuntimeError("boom")
        done.append(dataset)
        (workdir / "datasets" / modality / dataset / "converted").mkdir(parents=True)

    _make_raw(workdir, name="HRF")
    _make_raw(workdir, name="STARE")
    monkeypatch.setattr(pipeline, "convert_dataset", converter)
    with pytest.raises(PipelineError) as exc:
        pipeline.process_datasets(["HRF", "STARE"], "Fundus", convert=True, keep_raw_data=True)
    assert list(exc.value.failures) == ["HRF"]
    assert done == ["STARE"]


def test_download_that_produces_nothing_is_an_error(workdir, monkeypatch):
    monkeypatch.setattr(pipeline, "download_dataset", lambda *a, **k: None)
    with pytest.raises(PipelineError) as exc:
        pipeline.process_datasets(["HRF"], "Fundus", download=True)
    assert isinstance(exc.value.failures["HRF"], DownloadError)


def test_empty_download_files_do_not_count(workdir, monkeypatch):
    def fake_download(name, url, modality, **kwargs):
        d = workdir / "datasets" / modality / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "empty.zip").write_bytes(b"")

    monkeypatch.setattr(pipeline, "download_dataset", fake_download)
    with pytest.raises(PipelineError) as exc:
        pipeline.process_datasets(["HRF"], "Fundus", download=True)
    assert isinstance(exc.value.failures["HRF"], DownloadError)


def test_gdown_part_archive_counts_as_downloaded(workdir, monkeypatch):
    def fake_download(name, url, modality, **kwargs):
        d = workdir / "datasets" / modality / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "MSD_SPLEENabc123.part").write_bytes(b"complete tar archive")

    monkeypatch.setattr(pipeline, "download_dataset", fake_download)
    pipeline.process_datasets(["MSD_SPLEEN"], "CT", download=True)


def test_manual_dataset_without_files_explains_what_to_do(workdir, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "download_dataset", lambda *a, **k: called.append(a))
    with pytest.raises(PipelineError) as exc:
        pipeline.process_datasets(["BUSI"], "Ultrasound", download=True)
    err = exc.value.failures["BUSI"]
    assert isinstance(err, ManualDownloadRequired)
    assert "datasets/Ultrasound/BUSI/Dataset_BUSI.zip" in str(err)
    assert not called, "manual datasets must not hit the dead URL"


def test_manual_dataset_with_files_is_extracted(workdir, monkeypatch):
    ds = workdir / "datasets" / "Ultrasound" / "BUSI"
    ds.mkdir(parents=True)
    (ds / "Dataset_BUSI.zip").write_bytes(b"zip")
    extracted = []
    monkeypatch.setattr(pipeline, "extract_all_zips", lambda folder, name, **k: extracted.append(name))
    pipeline.process_datasets(["BUSI"], "Ultrasound", download=True)
    assert extracted == ["BUSI"]


def test_download_file_removes_truncated_file(tmp_path, monkeypatch):
    from omnimedseg.downloaders import download_datasets as dd

    class BrokenResponse:
        headers = {"content-type": "application/zip", "content-length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"partial"
            raise ConnectionError("connection reset")

    monkeypatch.setattr(dd.requests, "get", lambda *a, **k: BrokenResponse())
    dest = tmp_path / "file.zip"
    assert dd.download_file("http://example.invalid/file.zip", str(dest), "X") is False
    assert not dest.exists()

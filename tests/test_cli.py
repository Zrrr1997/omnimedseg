"""CLI behaviour: exit codes and modality handling (no network, no data)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run(*args, cwd):
    exe = shutil.which("omnimedseg", path=str(Path(sys.executable).parent))
    assert exe, "console script 'omnimedseg' not installed in this environment"
    return subprocess.run([exe, "--no-banner", *args], capture_output=True, text=True, cwd=cwd, timeout=300)


def test_unknown_dataset_exits_nonzero(tmp_path):
    assert _run("--dataset", "NOT_A_DATASET", cwd=tmp_path).returncode == 1


@pytest.mark.parametrize("modality", ["Ultrasound", "Ultrasound_2D"])
def test_ultrasound_modality_is_accepted(modality, tmp_path):
    result = _run("--modality", modality, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "modality: Ultrasound" in result.stderr + result.stdout


def test_manual_download_exits_nonzero_with_instructions(tmp_path):
    result = _run("--dataset", "BUSI", "--download", cwd=tmp_path)
    assert result.returncode == 1
    assert "Dataset_BUSI.zip" in result.stderr + result.stdout


def test_data_root_option(tmp_path):
    root = tmp_path / "my_data"
    result = _run("--data-root", str(root), "--dataset", "BUSI", "--download", cwd=tmp_path)
    out = result.stderr + result.stdout
    assert result.returncode == 1
    assert str(root / "Ultrasound" / "BUSI") in out
    assert f"Data root: {root}" in out

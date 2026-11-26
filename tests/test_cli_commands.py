"""The subcommand CLI: catalogue commands, selection, exit codes, and the legacy syntax."""

import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnimedseg.cli.legacy import is_legacy
from omnimedseg.core.cards import load_cards


def run(*args, cwd):
    exe = shutil.which("omnimedseg", path=str(Path(sys.executable).parent))
    assert exe
    return subprocess.run([exe, *args], capture_output=True, text=True, cwd=cwd, timeout=300)


def test_no_arguments_prints_help(tmp_path):
    r = run(cwd=tmp_path)
    assert r.returncode == 0 and "<command>" in r.stdout


def test_list_json_has_every_dataset_and_clean_stdout(tmp_path):
    r = run("list", "--format", "json", cwd=tmp_path)
    assert r.returncode == 0
    rows = json.loads(r.stdout)  # stdout must be pure JSON: no banner, no logs
    assert len(rows) == len(load_cards())
    assert {"name", "modality", "license", "commercial_use", "local", "radar_doi"} <= set(rows[0])


@pytest.mark.parametrize("flag,expected", [("commercial-ok", False), ("non-commercial", True), ("unknown", None)])
def test_list_license_filter(flag, expected, tmp_path):
    r = run("list", "--license", flag, "--format", "csv", cwd=tmp_path)
    names = [row["name"] for row in csv.DictReader(io.StringIO(r.stdout))]
    cards = load_cards()
    assert names and all(cards[n].license.non_commercial is expected for n in names)
    assert len(names) == sum(1 for c in cards.values() if c.license.non_commercial is expected)


def test_list_modality_and_alias(tmp_path):
    a = json.loads(run("list", "-m", "Ultrasound", "--format", "json", cwd=tmp_path).stdout)
    b = json.loads(run("list", "-m", "Ultrasound_2D", "--format", "json", cwd=tmp_path).stdout)
    assert a == b and {r["modality"] for r in a} == {"Ultrasound"}


def test_list_shows_local_status(tmp_path):
    (tmp_path / "root" / "Fundus" / "HRF" / "converted").mkdir(parents=True)
    rows = json.loads(run("list", "-m", "Fundus", "--format", "json", "--data-root", str(tmp_path / "root"),
                          cwd=tmp_path).stdout)
    status = {r["name"]: r["local"] for r in rows}
    assert status["HRF"] == "converted" and status["STARE"] == "-"


def test_info_is_case_insensitive(tmp_path):
    r = run("info", "hrf", cwd=tmp_path)
    assert r.returncode == 0 and "name: HRF" in r.stdout and "spdx: CC-BY-4.0" in r.stdout


def test_cite_writes_bibtex(tmp_path):
    out = tmp_path / "refs.bib"
    r = run("cite", "HRF", "ABDOMEN_1K", "-o", str(out), cwd=tmp_path)
    text = out.read_text()
    assert r.returncode == 0
    assert "marinov2026omnimedseg" in text and "budai2013robust" in text
    assert "ABDOMEN_1K" in r.stderr  # no citation recorded yet -> warned, not silently dropped


@pytest.mark.parametrize("args", [["info", "NOPE"], ["convert"], ["prepare", "NOPE"], ["clean", "HRF"]])
def test_user_errors_exit_1(args, tmp_path):
    assert run(*args, cwd=tmp_path).returncode == 1


def test_training_flags_are_gone(tmp_path):
    assert run("--train", cwd=tmp_path).returncode == 2


@pytest.mark.parametrize("order", ["before", "after"])
def test_data_root_before_or_after_command(order, tmp_path):
    root = tmp_path / "data"
    args = ["--data-root", str(root), "download", "BUSI"] if order == "before" else \
        ["download", "BUSI", "--data-root", str(root)]
    r = run(*args, cwd=tmp_path)
    assert r.returncode == 1  # manual download, archive not there
    assert str(root / "Ultrasound" / "BUSI") in r.stderr


def test_legacy_syntax_still_works_with_deprecation_note(tmp_path):
    r = run("--no-banner", "--dataset", "BUSI", "--download", cwd=tmp_path)
    assert r.returncode == 1
    assert "deprecated" in r.stderr and "omnimedseg download BUSI" in r.stderr


@pytest.mark.parametrize("argv,legacy", [
    (["--dataset", "HRF", "--download"], True),
    (["--modality", "CT", "--convert"], True),
    (["list", "--modality", "CT"], False),
    (["download", "HRF"], False),
    (["--data-root", "x", "list"], False),
    ([], False),
])
def test_is_legacy(argv, legacy):
    assert is_legacy(argv) is legacy

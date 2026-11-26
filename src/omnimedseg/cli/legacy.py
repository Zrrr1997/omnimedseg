"""The pre-subcommand CLI (`omnimedseg --dataset X --download --convert`), kept for compatibility.

Deprecated: prints a note pointing to the equivalent subcommand. Training flags were removed.
"""

import argparse
import logging
import sys
from typing import List

from omnimedseg.cli.commands import MODALITY_ALIASES, MODALITY_CHOICES, CLIError, run_pipeline, select_cards
from omnimedseg.core.paths import set_data_root

logger = logging.getLogger(__name__)

LEGACY_FLAGS = {
    "--dataset", "--modality", "--download", "--extract", "--convert", "--check_integrity",
    "--visualization", "--force_convert", "--keep_raw_data", "--delete_raw_data", "--filter_empty_labels",
}


SUBCOMMANDS = {"list", "info", "cite", "prepare", "download", "extract", "convert", "verify", "clean", "browse"}


def is_legacy(argv: List[str]) -> bool:
    """Old syntax = no subcommand anywhere, but at least one of the old action/selection flags."""
    if any(a in SUBCOMMANDS for a in argv):
        return False
    return any(a.split("=", 1)[0] in LEGACY_FLAGS for a in argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omnimedseg", description="(deprecated flag syntax; see `omnimedseg --help`)")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--modality", type=str, default="Fundus", choices=MODALITY_CHOICES)
    p.add_argument("--data-root", "--data_root", dest="data_root", default=None)
    for flag in ("--download", "--extract", "--convert", "--check_integrity", "--visualization",
                 "--force_convert", "--keep_raw_data", "--delete_raw_data", "--filter_empty_labels",
                 "--verbose", "--quiet", "--no-banner", "--debug"):
        p.add_argument(flag, action="store_true")
    # dataset browser options (with --visualization)
    p.add_argument("--root", type=str, default="datasets_to_review")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--max_images", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    return p


def _suggestion(a) -> str:
    if a.visualization:
        return "omnimedseg browse"
    target = a.dataset or f"--modality {a.modality}"
    if a.download and a.convert:
        return f"omnimedseg prepare {target}"
    for flag, cmd in (("download", "download"), ("extract", "extract"), ("convert", "convert"),
                      ("check_integrity", "verify"), ("delete_raw_data", "clean --raw")):
        if getattr(a, flag):
            return f"omnimedseg {cmd} {target}"
    return "omnimedseg --help"


def main(argv: List[str]) -> int:
    a = build_parser().parse_args(argv)
    a.modality = MODALITY_ALIASES.get(a.modality, a.modality)
    if a.data_root:
        set_data_root(a.data_root)
    print(f"ℹ️  This flag syntax is deprecated. Equivalent: `{_suggestion(a)}`", file=sys.stderr)

    if a.visualization:
        from omnimedseg.utils.visualize import start_visualization

        start_visualization(root=a.root, size=a.size, host=a.host, port=a.port, debug=a.debug,
                            max_images=a.max_images, workers=a.workers)
        return 0

    try:
        cards = select_cards([a.dataset]) if a.dataset else select_cards([], [a.modality])
    except CLIError as e:
        logger.error(str(e))
        return 1
    return run_pipeline(
        cards,
        download=a.download,
        convert=a.convert,
        extract=a.extract,
        delete_raw_data=a.delete_raw_data,
        force_convert=a.force_convert,
        keep_raw_data=a.keep_raw_data,
        filter_empty=a.filter_empty_labels,
        check_integrity=a.check_integrity,
    )

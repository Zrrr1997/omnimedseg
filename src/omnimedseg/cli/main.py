#!/usr/bin/env python
"""OmniMedSeg command-line interface.

    omnimedseg list [--modality CT] [--license commercial-ok]   # browse the catalogue
    omnimedseg info HRF                                          # one dataset card
    omnimedseg prepare HRF KVASIR_SEG                            # download + convert
    omnimedseg cite HRF KVASIR_SEG -o refs.bib                   # BibTeX for what you used

Run `omnimedseg <command> --help` for details. The old flag syntax
(`omnimedseg --dataset X --download --convert`) still works but is deprecated.
"""

import argparse
import logging
import sys
from typing import List, Optional

from omnimedseg.cli import commands
from omnimedseg.cli.commands import MODALITY_CHOICES, CLIError
from omnimedseg.core.cards import CardError, load_cards
from omnimedseg.core.paths import set_data_root
from omnimedseg.utils.banner import print_banner
from omnimedseg.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# Commands whose stdout is data (no banner, quiet logs).
_CATALOGUE_COMMANDS = {"list", "info", "cite"}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("omnimedseg")
    except Exception:
        return "unknown"


def _common_options(parser: argparse.ArgumentParser, suppress_defaults: bool) -> None:
    """Options accepted both before and after the subcommand."""
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--data-root", "--data_root", dest="data_root", default=default, metavar="DIR",
                        help="where datasets are stored (default: $OMNIMEDSEG_DATA_ROOT, else ./datasets)")
    parser.add_argument("-v", "--verbose", action="store_true", default=default if suppress_defaults else False,
                        help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", default=default if suppress_defaults else False,
                        help="only log errors")
    parser.add_argument("--no-banner", action="store_true", default=default if suppress_defaults else False,
                        help="do not print the ASCII banner")


def _selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help="dataset name(s), e.g. HRF KVASIR_SEG")
    parser.add_argument("-m", "--modality", action="append", choices=MODALITY_CHOICES, metavar="MODALITY",
                        help="all datasets of a modality (repeatable): " + ", ".join(MODALITY_CHOICES))
    parser.add_argument("--all", action="store_true", help="all datasets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnimedseg",
        description=f"Download, convert and explore {len(load_cards())} standardized medical image "
                    "segmentation datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  omnimedseg list --modality Fundus --license commercial-ok
  omnimedseg info KVASIR_SEG
  omnimedseg prepare HRF KVASIR_SEG --data-root /scratch/omnimedseg
  omnimedseg download --modality Fundus
  omnimedseg convert HRF --force --keep-raw
  omnimedseg cite HRF KVASIR_SEG -o references.bib
  omnimedseg browse --port 8080""",
    )
    parser.add_argument("--version", action="version", version=f"omnimedseg {_version()}")
    _common_options(parser, suppress_defaults=False)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name, help_text, func):
        p = sub.add_parser(name, help=help_text, description=help_text)
        _common_options(p, suppress_defaults=True)
        p.set_defaults(func=func)
        return p

    p = add("list", "list datasets with license and local status", commands.cmd_list)
    p.add_argument("-m", "--modality", action="append", choices=MODALITY_CHOICES, metavar="MODALITY",
                   help="only these modalities (repeatable)")
    p.add_argument("--license", choices=sorted(commands.LICENSE_FILTERS),
                   help="filter by commercial-use terms of the dataset license")
    p.add_argument("--format", choices=["table", "csv", "json"], default="table")

    p = add("info", "show the dataset card(s) and local status", commands.cmd_info)
    p.add_argument("datasets", nargs="+", metavar="DATASET")

    p = add("cite", "print BibTeX for the selected datasets (plus OmniMedSeg)", commands.cmd_cite)
    _selection(p)
    p.add_argument("-o", "--output", metavar="FILE", help="write to a .bib file instead of stdout")
    p.add_argument("--no-omnimedseg", action="store_true", help="do not include the OmniMedSeg citation")

    def conversion_options(p):
        p.add_argument("--force", action="store_true", help="convert again even if already converted")
        p.add_argument("--keep-raw", action="store_true", help="keep extracted raw data after conversion")
        p.add_argument("--filter-empty-labels", action="store_true",
                       help="drop samples whose labels are all empty")

    p = add("prepare", "download and convert (the usual way to get a dataset)", commands.cmd_prepare)
    _selection(p)
    conversion_options(p)

    p = add("download", "download (and, for most sources, extract) datasets", commands.cmd_download)
    _selection(p)

    p = add("extract", "extract already downloaded archives", commands.cmd_extract)
    _selection(p)

    p = add("convert", "convert downloaded datasets to the standard format", commands.cmd_convert)
    _selection(p)
    conversion_options(p)

    p = add("verify", "check that converted files and their sources exist", commands.cmd_verify)
    _selection(p)

    p = add("clean", "delete raw data and/or samples with empty labels", commands.cmd_clean)
    _selection(p)
    p.add_argument("--raw", action="store_true", help="delete extracted raw data")
    p.add_argument("--empty-labels", action="store_true", help="remove converted samples with empty labels")

    p = add("browse", "start the web-based dataset browser", commands.cmd_browse)
    p.add_argument("--root", default=None, help="folder to browse (default: the data root)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--size", type=int, default=512, help="display size in pixels")
    p.add_argument("--max-images", type=int, default=10, help="images loaded per dataset")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--debug", action="store_true")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    from omnimedseg.cli import legacy

    if legacy.is_legacy(argv):
        a = legacy.build_parser().parse_known_args(argv)[0]
        if not a.no_banner:
            print_banner()
        setup_logging(level=logging.ERROR if a.quiet else logging.INFO, verbose=a.verbose)
        sys.exit(legacy.main(argv))

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)

    catalogue = args.command in _CATALOGUE_COMMANDS
    if not (args.no_banner or catalogue):
        print_banner()
    level = logging.ERROR if (args.quiet or (catalogue and not args.verbose)) else logging.INFO
    setup_logging(level=level, verbose=args.verbose)
    if args.data_root:
        set_data_root(args.data_root)

    try:
        code = args.func(args)
    except (CLIError, CardError) as e:
        logger.error(str(e))
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()

"""Implementations of the `omnimedseg <command>` subcommands.

Each ``cmd_*`` function takes the parsed argparse namespace and returns a process exit code.
Command output goes to stdout; logs and progress go to stderr.
"""

import csv
import json
import logging
import os
import sys
from collections import OrderedDict
from typing import Dict, List

import yaml

from omnimedseg.core.cards import MODALITIES, DatasetCard, load_cards
from omnimedseg.core.errors import PipelineError
from omnimedseg.core.paths import data_root, dataset_dir
from omnimedseg.core.pipeline import process_datasets

logger = logging.getLogger(__name__)

# Older modality names still accepted on the command line.
MODALITY_ALIASES = {"Ultrasound_2D": "Ultrasound"}
MODALITY_CHOICES = sorted(MODALITIES) + sorted(MODALITY_ALIASES)

OMNIMEDSEG_BIBTEX = """@misc{marinov2026omnimedseg,
  title={OmniMedSeg: A Large-Scale Standardized Benchmark for Interactive Medical Image Segmentation},
  author={Marinov, Zdravko and Jaus, Alexander and Rei{\\ss}, Simon and Schlumberger, Tobias and Wei, Jiale and Wen, Di and Zheng, Junwei and Kleesiek, Jens and Stiefelhagen, Rainer},
  year={2026},
  publisher={KIT Library},
  doi={10.5445/IR/1000194202}
}"""


def _n(count: int, word: str = "dataset") -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


class CLIError(Exception):
    """A user error (unknown dataset, bad selection) reported without a traceback."""


# ---------------------------------------------------------------------------
# Selecting datasets
# ---------------------------------------------------------------------------

def _canonical_modality(m: str) -> str:
    return MODALITY_ALIASES.get(m, m)


def select_cards(names: List[str], modalities: List[str] = None, all_datasets: bool = False) -> List[DatasetCard]:
    """Resolve positional dataset names and/or --modality/--all into cards (order preserved, no duplicates)."""
    cards = load_cards()
    selected: "OrderedDict[str, DatasetCard]" = OrderedDict()
    unknown = []
    for name in names or []:
        key = name.upper()
        if key in cards:
            selected[key] = cards[key]
        else:
            unknown.append(name)
    if unknown:
        raise CLIError(f"unknown dataset: {', '.join(unknown)} (run `omnimedseg list` to see all datasets)")
    for m in modalities or []:
        m = _canonical_modality(m)
        for name, card in cards.items():
            if card.modality == m:
                selected.setdefault(name, card)
    if all_datasets:
        for name, card in cards.items():
            selected.setdefault(name, card)
    return list(selected.values())


def _license_label(card: DatasetCard) -> str:
    if card.license.spdx:
        return card.license.spdx
    return "other" if card.license.text else "unknown"


def _commercial(card: DatasetCard) -> str:
    nc = card.license.non_commercial
    return "?" if nc is None else ("no" if nc else "yes")


def _local_status(card: DatasetCard) -> str:
    d = dataset_dir(card.modality, card.name)
    if os.path.isdir(os.path.join(d, "converted")):
        return "converted"
    if os.path.isdir(os.path.join(d, "converted.failed")):
        return "failed"
    if os.path.isdir(d) and any(os.scandir(d)):
        return "downloaded"
    return "-"


# ---------------------------------------------------------------------------
# Catalogue commands: list / info / cite
# ---------------------------------------------------------------------------

LICENSE_FILTERS = {
    "commercial-ok": lambda c: c.license.non_commercial is False,
    "non-commercial": lambda c: c.license.non_commercial is True,
    "unknown": lambda c: c.license.non_commercial is None,
}


def cmd_list(args) -> int:
    cards = select_cards([], args.modality, all_datasets=not args.modality)
    if args.license:
        cards = [c for c in cards if LICENSE_FILTERS[args.license](c)]

    rows = [
        OrderedDict(
            name=c.name,
            modality=c.modality,
            license=_license_label(c),
            commercial_use=_commercial(c),
            download=c.download.method,
            local=_local_status(c),
            radar_doi=c.radar_doi or "",
        )
        for c in cards
    ]
    out = sys.stdout
    if args.format == "json":
        json.dump(rows, out, indent=1)
        out.write("\n")
    elif args.format == "csv":
        if rows:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        cols = ["name", "modality", "license", "commercial_use", "download", "local"]
        headers = {"name": "NAME", "modality": "MODALITY", "license": "LICENSE",
                   "commercial_use": "COMMERCIAL", "download": "DOWNLOAD", "local": "LOCAL"}
        widths = {k: max([len(headers[k])] + [len(str(r[k])) for r in rows]) for k in cols}
        print("  ".join(headers[k].ljust(widths[k]) for k in cols), file=out)
        for r in rows:
            print("  ".join(str(r[k]).ljust(widths[k]) for k in cols), file=out)
        print(f"\n{_n(len(rows))}. Data root: {os.path.abspath(data_root())}", file=sys.stderr)
    return 0


def cmd_info(args) -> int:
    cards = select_cards(args.datasets)
    for i, card in enumerate(cards):
        if i:
            print("---")
        data = card.model_dump()
        data["local"] = {"status": _local_status(card), "path": os.path.abspath(dataset_dir(card.modality, card.name))}
        print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100).rstrip())
    return 0


def cmd_cite(args) -> int:
    cards = select_cards(args.datasets, args.modality, args.all)
    if not cards:
        raise CLIError("Nothing to cite: give dataset names, --modality, or --all.")
    entries = [] if args.no_omnimedseg else [OMNIMEDSEG_BIBTEX]
    missing = []
    for card in cards:
        if card.source.citation:
            entries.append(f"% {card.name} ({card.modality})\n{card.source.citation.strip()}")
        else:
            missing.append(card.name)
    text = "\n\n".join(entries) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {len(entries)} BibTeX entries to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    if missing:
        print(f"warning: no citation recorded for {', '.join(missing)}; see `omnimedseg info <name>` for the homepage",
              file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------

def run_pipeline(cards: List[DatasetCard], **steps) -> int:
    """Run the pipeline on cards (grouped per modality) and report failures. Returns an exit code."""
    if not cards:
        raise CLIError("No datasets selected: give dataset names, --modality, or --all.")
    by_modality: Dict[str, List[str]] = OrderedDict()
    for c in cards:
        by_modality.setdefault(c.modality, []).append(c.name)

    logger.info(f"Data root: {os.path.abspath(data_root())}")
    logger.info(f"Datasets:  {', '.join(c.name for c in cards[:8])}"
                + (f" and {len(cards) - 8} more" if len(cards) > 8 else ""))
    failures = {}
    try:
        for modality, names in by_modality.items():
            try:
                process_datasets(names, modality, **steps)
            except PipelineError as e:
                failures.update(e.failures)
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130

    if failures:
        logger.error(f"{len(failures)} of {_n(len(cards))} failed:")
        for name, err in failures.items():
            print(f"  {name}: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    logger.info(f"\nDone ({_n(len(cards))}).")
    return 0


def _selected(args) -> List[DatasetCard]:
    return select_cards(args.datasets, args.modality, args.all)


def cmd_download(args) -> int:
    return run_pipeline(_selected(args), download=True)


def cmd_extract(args) -> int:
    return run_pipeline(_selected(args), extract=True)


def cmd_convert(args) -> int:
    return run_pipeline(_selected(args), convert=True, force_convert=args.force,
                        keep_raw_data=args.keep_raw, filter_empty=args.filter_empty_labels)


def cmd_prepare(args) -> int:
    return run_pipeline(_selected(args), download=True, convert=True, force_convert=args.force,
                        keep_raw_data=args.keep_raw, filter_empty=args.filter_empty_labels)


def cmd_verify(args) -> int:
    return run_pipeline(_selected(args), check_integrity=True)


def cmd_clean(args) -> int:
    if not (args.raw or args.empty_labels):
        raise CLIError("Nothing to clean: pass --raw and/or --empty-labels.")
    return run_pipeline(_selected(args), delete_raw_data=args.raw, filter_empty=args.empty_labels)


def cmd_browse(args) -> int:
    from omnimedseg.utils.visualize import start_visualization

    start_visualization(
        root=args.root or data_root(),
        size=args.size,
        host=args.host,
        port=args.port,
        debug=args.debug,
        max_images=args.max_images,
        workers=args.workers,
    )
    return 0

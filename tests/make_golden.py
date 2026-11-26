"""Record golden checksums from the currently converted datasets.

Only run this on output produced by a trusted version of the code:

    python tests/make_golden.py            # all golden datasets
    python tests/make_golden.py HRF        # just one
"""

import json
import sys

from golden_utils import GOLDEN_DATASETS, GOLDEN_DIR, REPO_ROOT, golden_path, hash_converted


def main(selected):
    GOLDEN_DIR.mkdir(exist_ok=True)
    for modality, dataset in GOLDEN_DATASETS:
        if selected and dataset not in selected:
            continue
        converted = REPO_ROOT / "datasets" / modality / dataset / "converted"
        if not converted.is_dir():
            print(f"skip {dataset}: {converted} does not exist")
            continue
        hashes = hash_converted(converted)
        golden_path(dataset).write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n")
        print(f"wrote {golden_path(dataset)} ({len(hashes)} files)")


if __name__ == "__main__":
    main(set(sys.argv[1:]))

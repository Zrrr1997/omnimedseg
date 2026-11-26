"""Hash converted datasets so refactors can be checked for byte-for-byte identical output.

Images are hashed on *decoded* content (pixels/voxels + shape + dtype, and the affine for NIfTI),
so different PNG/gzip encoder versions do not cause false alarms.
"""

import hashlib
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# (modality folder, dataset name) — small datasets covering 2D grayscale-label, 2D RGB, and 3D CT paths.
GOLDEN_DATASETS = [
    ("Fundus", "HRF"),
    ("Endoscopy", "KVASIR_SEG"),
    ("CT", "MSD_SPLEEN"),
]


def _hash_array(arr: np.ndarray, extra: bytes = b"") -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(extra)
    return h.hexdigest()


def hash_file(path: Path) -> str:
    name = path.name
    if name.endswith(".png"):
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not read {path}")
        return _hash_array(arr)
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        img = nib.load(str(path))
        affine = np.round(img.affine, 4).tobytes()
        return _hash_array(np.asanyarray(img.dataobj), extra=affine)
    if name.endswith(".json"):
        return hashlib.sha256(json.dumps(json.loads(path.read_text()), sort_keys=True).encode()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_converted(converted_dir: Path) -> dict:
    """Return {relative_path: hash} for every file below a dataset's converted/ folder."""
    files = sorted(p for p in converted_dir.rglob("*") if p.is_file())
    return {str(p.relative_to(converted_dir)): hash_file(p) for p in files}


def golden_path(dataset: str) -> Path:
    return GOLDEN_DIR / f"{dataset}.json"

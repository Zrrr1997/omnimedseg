"""Faster parallel downloader for KiTS23 dataset"""
import sys
import math
import time
from pathlib import Path
from shutil import move
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


TRAINING_CASE_NUMBERS = list(range(300)) + list(range(400, 589))

# Destination path inside extracted structure
DST_PTH = (
    Path("datasets")
    / "CT"
    / "KITS23"
    / "extracted"
    / "dataset"
    / "kits23-main"
)

# Tuning parameters
MAX_WORKERS = 8          # Number of parallel downloads
CHUNK_SIZE = 1024 * 1024 # 1 MB chunks


def get_destination(case_id: str, create: bool = False) -> Path:
    destination = DST_PTH / case_id / "imaging.nii.gz"
    if create:
        destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def download_case(case_num: int, session: requests.Session) -> str:
    """Download one case with retries and streaming"""
    remote_name = f"master_{case_num:05d}.nii.gz"
    url = f"https://kits19.sfo2.digitaloceanspaces.com/{remote_name}"
    destination = get_destination(f"case_{case_num:05d}", True)
    tmp_pth = destination.parent / f".partial.{destination.name}"

    tries = 0
    while True:
        try:
            tries += 1
            with session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                total_chunks = math.ceil(total_size / CHUNK_SIZE)

                with tmp_pth.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            move(str(tmp_pth), str(destination))
            return f"✅ case_{case_num:05d} done"
        except Exception as e:
            if tries < 5:
                print(f"Retry case_{case_num:05d} (attempt {tries}) in 30s: {e}")
                time.sleep(30)
            else:
                return f"❌ case_{case_num:05d} failed: {e}"


def download_dataset():
    DST_PTH.mkdir(parents=True, exist_ok=True)

    left_to_download = [
        case_num
        for case_num in TRAINING_CASE_NUMBERS
        if not get_destination(f"case_{case_num:05d}").exists()
    ]

    print(f"\nFound {len(left_to_download)} cases to download\n")

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_case, case_num, session): case_num
                for case_num in left_to_download
            }

            for f in tqdm(as_completed(futures), total=len(futures), desc="Dataset"):
                result = f.result()
                print(result)


if __name__ == "__main__":
    try:
        download_dataset()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.\n")
        sys.exit(1)

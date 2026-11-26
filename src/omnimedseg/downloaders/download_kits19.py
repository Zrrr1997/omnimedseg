from pathlib import Path
from shutil import move
import os
import sys
import time
import math
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

imaging_url = "https://kits19.sfo2.digitaloceanspaces.com/"
imaging_name_tmplt = "master_{:05d}.nii.gz"
temp_f = Path(__file__).parent / "temp.tmp"

MAX_WORKERS = 8          # parallel downloads
CHUNK_SIZE = 1024 * 1024 # 1 MB


def get_destination(i):
    """Build destination path for case i"""
    destination = Path(os.environ.get("OMNIMEDSEG_DATA_ROOT", "datasets")) / "CT" / "KITS19" / "extracted" / "kits19-master" / \
                  "data" / f"case_{i:05d}" / "imaging.nii.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def download_case(cid, session):
    """Download one case with retries"""
    destination = get_destination(cid)
    remote_name = imaging_name_tmplt.format(cid)
    uri = imaging_url + remote_name

    tries = 0
    while True:
        try:
            tries += 1
            with session.get(uri, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                total_chunks = math.ceil(total_size / CHUNK_SIZE)

                temp_path = temp_f.with_name(f"temp_{cid}.tmp")
                with temp_path.open("wb") as f, tqdm(
                    unit="MB",
                    unit_scale=True,
                    desc=f"case_{cid:05d}",
                    total=total_chunks,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(1)

                move(str(temp_path), str(destination))
            return f"Finished case {cid}"
        except Exception as e:
            if tries < 5:
                print(f"Retry {cid} (attempt {tries}) in 30s: {e}")
                time.sleep(30)
            else:
                return f"Failed case {cid}: {e}"


if __name__ == "__main__":
    left_to_download = [
        i for i in range(210) if not get_destination(i).exists()
    ]

    print(f"{len(left_to_download)} cases to download...")

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(download_case, cid, session) for cid in left_to_download]

            for future in as_completed(futures):
                print(future.result())

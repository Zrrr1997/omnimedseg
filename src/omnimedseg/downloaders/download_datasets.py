import os
from urllib.parse import urlparse
from tqdm import tqdm
from omnimedseg.utils.postprocess_download import unzip_file, is_compressed_file, is_unzipped
import requests
import gdown
import shutil
import zipfile
import io
import subprocess
import urllib3
from tcia_utils import nbia
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import re
import time
import undetected_chromedriver as uc

FIGSHARE_API_URL = 'https://api.figshare.com/v2'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def download_file(url, dest_path, name, user_agent=None, verify_ssl=True, part=''):
    try:
        print(f"📥 Downloading {name} → {dest_path}")

        headers = {
            'User-Agent': user_agent or 'Mozilla/5.0',
            'Referer': 'https://data.mendeley.com/',  # Some sites care about this
        }

        # Ensure dest_path is a file path
        if os.path.isdir(dest_path) or dest_path.endswith(os.sep):
            os.makedirs(dest_path, exist_ok=True)
            dest_path = os.path.join(dest_path, f"{name}.zip")

        # Make the request
        with requests.get(url, headers=headers, stream=True, allow_redirects=True, verify=verify_ssl) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")

            if "text/html" in content_type:
                raise RuntimeError(
                    f"Received HTML instead of binary file from {url}"
                )
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=name)

            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            pbar.close()

        print(f"✅ {name} downloaded{part}.")
        if '.tcia' in dest_path:
            handle_tcia(name, dest_path, os.path.dirname(dest_path))
        return True

    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")
        # Remove a truncated file, otherwise already_downloaded() would skip it on every later run.
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        return False


def download_gdown(url, dest_path, name=None):
    """
    Download a file from Google Drive using gdown with tqdm progress.
    Args:
        url (str): The Google Drive shareable URL.
        dest_path (str): Path to save the downloaded file.
        name (str): Optional display name for progress.
    """
    try:
        print(f"📥 Downloading {name or os.path.basename(dest_path)} from Google Drive → {dest_path}")
        if 'folder' not in url:
            gdown.download(url, output=dest_path, quiet=False, fuzzy=True)
        else:
            gdown.download_folder(id=url.split('/')[-1], output=dest_path, quiet=False, use_cookies=False)

        print(f"✅ {name or os.path.basename(dest_path)} downloaded.")
    except Exception as e:
        print(f"❌ Error downloading from Google Drive: {e}")


def handle_figshare(name, url, dest_folder):
    """
    Handle downloading from Figshare using the official API.
    Uses the /articles/{id}/files endpoint to get file info,
    then downloads using the file download endpoint.
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    # Check if already downloaded
    if already_downloaded(dest_folder):
        print(f"✅ {name} already exists at {dest_folder}. Skipping download.")
        return
    
    print(f"📥 Downloading {name} from Figshare → {dest_folder}")
    
    # Normalize URL to list
    if isinstance(url, str):
        urls = [url]
    else:
        urls = url
    
    # Figshare API base URL
    BASE_URL = 'https://api.figshare.com/v2'
    
    for idx, single_url in enumerate(urls):
        # Extract article ID from URL
        article_id = single_url.strip().split('/')[-1]
        # Step 1: Get file metadata from API
        print(f"🔍 Fetching file info from Figshare API for article {article_id}...")
        r = requests.get(f'{BASE_URL}/articles/{article_id}/files')
        
        if r.status_code != 200:
            print(f'❌ Failed to get file info: {r.content}')
            continue
        
        file_metadata = r.json()
        
        # Step 2: Download each file
        for file_info in file_metadata:
            file_id = file_info['id']
            file_name = file_info['name']
            
            # Determine destination path
            if len(file_metadata) > 1:
                dest_path = os.path.join(dest_folder, file_name)
            else:
                dest_path = os.path.join(dest_folder, f"{name}_{idx}.zip")
            
            # Skip if already downloaded
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                print(f"⏭️  {file_name} already exists, skipping")
                continue
            
            # Step 3: Download the file using the file download endpoint
            print(f"📥 Downloading {file_name}...")
            response = requests.get(
                f'{BASE_URL}/file/download/{file_id}',
                stream=True
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to download {file_name}: {response.status_code}")
                continue
            
            # Save with progress bar
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Downloaded {file_name}")
    
    print(f"✅ All Figshare files for {name} downloaded.")


def download_github_subdir(repo_url, subdir_path, dest_dir):
    """
    Downloads a specific subdirectory from a GitHub repository.

    Args:
        repo_url (str): URL of the GitHub repository (e.g., 'https://github.com/user/repo')
        subdir_path (str): Path to the subdirectory within the repository (e.g., 'folder/subfolder')
        dest_dir (str): Local directory to save the downloaded subdirectory
    """
    # Extract user and repo name from the URL
    parts = repo_url.rstrip('/').split('/')
    user, repo = parts[-2], parts[-1]

    # Construct the URL to download the repository as a ZIP file
    zip_url = f'https://github.com/{user}/{repo}/archive/refs/heads/main.zip'

    # Download the ZIP file
    print(f"Downloading repository ZIP from {zip_url}...")
    response = requests.get(zip_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download ZIP file: {response.status_code}")

    # Extract the specific subdirectory
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        # Construct the path prefix inside the ZIP file
        zip_subdir_prefix = f"{repo}-main/{subdir_path}/"

        # List of files to extract
        members = [m for m in zip_ref.namelist() if m.startswith(zip_subdir_prefix)]
        if not members:
            raise Exception(f"Subdirectory '{subdir_path}' not found in the repository.")

        # Extract files to a temporary directory
        temp_extract_dir = os.path.join(dest_dir, '__temp_extract__')
        zip_ref.extractall(temp_extract_dir, members)

        # Move the extracted subdirectory to the destination directory
        extracted_subdir = os.path.join(temp_extract_dir, zip_subdir_prefix)
        final_dest = os.path.join(dest_dir, os.path.basename(subdir_path))
        if os.path.exists(final_dest):
            shutil.rmtree(final_dest)
        shutil.move(extracted_subdir, final_dest)

        # Clean up temporary directory
        shutil.rmtree(temp_extract_dir)

    print(f"Subdirectory '{subdir_path}' has been downloaded to '{final_dest}'.")

    zip_filename = f"{final_dest}.zip"
    shutil.make_archive(final_dest, 'zip', final_dest)
    print(f"🗜️ Zipped '{final_dest}' to '{zip_filename}'.")
    shutil.move(final_dest, os.path.join(os.path.dirname(final_dest), 'extracted'))


def download_github_repo(repo_url, dest_dir):
    """
    Downloads an entire GitHub repository.
    """
    parts = repo_url.rstrip('/').split('/')
    user, repo = parts[-2], parts[-1]
    
    zip_url = f'https://github.com/{user}/{repo}/archive/refs/heads/main.zip'
    
    print(f"Downloading repository ZIP from {zip_url}...")
    response = requests.get(zip_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download ZIP file: {response.status_code}")
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(dest_dir)
    
    print(f"Repository '{repo}' downloaded to '{dest_dir}'.")


def extract_all_zips(dest_folder, name, filter_pattern=None, new_folder=False):
    zip_files = [f for f in os.listdir(dest_folder) if is_compressed_file(os.path.join(dest_folder, f))]
    
    print("🧾 Found zip/rar/tar files:", zip_files)
    if len(zip_files) == 0:
        print(f"⚠️  There are no found archives for {name}!")
        all_files = [os.path.join(dest_folder, f) for f in os.listdir(dest_folder) if '.mat' in f]
        if len(all_files) > 0:
            os.makedirs(os.path.join(dest_folder, 'extracted'))
            print(f'Found {len(all_files)} .mat files! Moving them to {dest_folder}/extracted')
            for file in all_files:
                shutil.copy(file, os.path.join(dest_folder, 'extracted', os.path.basename(file)))


    extract_path = os.path.join(dest_folder, 'extracted')

    for zip_file in zip_files:
        zip_path = os.path.join(dest_folder, zip_file)
        if is_unzipped(extract_path, zip_path):
            print(f"✅ {zip_file} from {name} already unzipped.")
        else:
            filter_pattern_ = None if 'CHAKSU' in dest_folder else filter_pattern
            if new_folder and name != 'VESSEL12' and name != 'MEDSEG_VENTRICLES' and name != 'PROMISE12':
                extract_path_ = os.path.join(extract_path, os.path.basename(zip_file).split('.')[0])
                os.makedirs(extract_path_, exist_ok=True)
                unzip_file(zip_path, extract_to=extract_path_, filter_pattern=filter_pattern_)
            else:
                os.makedirs(extract_path, exist_ok=True)
                unzip_file(zip_path, extract_to=extract_path, filter_pattern=filter_pattern_)

    # Handle inner zip/rar files too
    inner_zip_files = [f for f in os.listdir(extract_path) if is_compressed_file(os.path.join(extract_path, f))]
    for zip_file in inner_zip_files:
        zip_path = os.path.join(extract_path, zip_file)
        if is_unzipped(extract_path, zip_path, filter_pattern=filter_pattern):
            print(f"✅ {zip_file} from {name} already unzipped.")
        else:
            if new_folder:
                extract_path_ = os.path.join(extract_path, os.path.basename(zip_file).split('.')[0])
                unzip_file(zip_path, extract_to=extract_path_, filter_pattern=filter_pattern)
            else:
                unzip_file(zip_path, extract_to=extract_path, filter_pattern=filter_pattern)


def get_source(url):
    if isinstance(url, list):
        url = url[0]

    source_map = {
        "figshare": "figshare",
        'rdr': "figshare",
        "zenodo": "zenodo",
        "mendeley": "mendeley",
        "kingston": "kingston",
        "prod-dcd": "prod-dcd",
        "drive.google": "gdown",
        "github": "github",
        "researchgate": "researchgate",
        "cancerimagingarchive.net/collection": "TCIA",
        ".tcia": "TCIA",
        "dataverse": "dataverse",   
    }

    for key, value in source_map.items():
        if key in url:
            return value

    return "custom"


def handle_dataverse(name, url, dest_folder):
    """
    Handle downloading datasets from Dataverse.
    Supports multi-file datasets with multithreading.
    """
    os.makedirs(dest_folder, exist_ok=True)

    if already_downloaded(dest_folder):
        print(f"✅ {name} already exists at {dest_folder}. Skipping download.")
        return

    print(f"📥 Downloading {name} from Dataverse → {dest_folder}")

    # In Dataverse, appending "?format=original" often gives direct file access
    if isinstance(url, str):
        urls = [url]
    else:
        urls = url

    def download_dataverse_file(file_url):
        try:
            filename = os.path.basename(urlparse(file_url).path)
            if not filename:
                filename = f"{name}.zip"

            dest_path = os.path.join(dest_folder, filename)
            if os.path.exists(dest_path):
                tqdm.write(f"⏭️ Skipping {filename}, already exists.")
                return

            with requests.get(file_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=filename)

                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                pbar.close()
            tqdm.write(f"✅ Downloaded {filename}")
        except Exception as e:
            tqdm.write(f"❌ Error downloading {file_url}: {e}")

    num_workers = min(multiprocessing.cpu_count(), 8)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(download_dataverse_file, urls), total=len(urls), desc="Files"))

    print(f"✅ All Dataverse files for {name} downloaded.")


def already_downloaded(dest_folder):
    condition_1 = os.path.exists(dest_folder)
    condition_2 = True
    if os.path.isdir(dest_folder):
        condition_2 = len(os.listdir(dest_folder)) > 0

    return (condition_1 and condition_2) 


def handle_gdown(name, url, dest_folder):
    if already_downloaded(dest_folder):
        print(f"✅ {name} already exists at {dest_folder}. Skipping download.")
    else:
        if isinstance(url, str):
            download_gdown(url, dest_folder, name)
        elif isinstance(url, list):
            for url_ in url:
                download_gdown(url_, dest_folder, name)


def handle_orvs(name, dest_folder, url):
    if already_downloaded(dest_folder):
        print(f"✅ {name} already exists at {dest_folder}. Skipping download.")
    else:
        if name == 'ORVS':
            download_github_subdir(
                repo_url='https://github.com/AbdullahSarhan/ICPRVessels',
                subdir_path='Vessels-Datasets/ORVS',
                dest_dir=dest_folder
            )


def handle_tcia(name, url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    print(f"📥  Downloading {name}...")

    # Check if URL is a .tcia manifest file
    if url.endswith('.tcia'):
        if 'datasets' in url:
            file_path = url
        else:
            file_name = url.split('/')[-1]
            file_path = os.path.join(dest_folder, file_name)

            # Download the .tcia manifest file
            print(f"Detected .tcia manifest file. Downloading {file_name} ...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Manifest downloaded to {file_path}")
            else:
                raise Exception(f"Failed to download manifest: HTTP {response.status_code}")

        # Parse manifest to get series UIDs
        with open(file_path, 'r') as f:
            series_uids = [line.strip() for line in f if '1.' in line]

    else:
        # Assume this is a TCIA collection name
        collection_name = url.rstrip('/').split('/')[-1]
        print(f"Fetching series list from collection: {collection_name} ...")
        if 'ct-colonography' in collection_name:
            collection_name = 'ct colonography'
        series_list = nbia.getSeries(collection_name)
        series_uids = [s['SeriesInstanceUID'] for s in series_list]

    def download_series(series_uid):
        try:

            os.makedirs(os.path.join(dest_folder, 'compressed', f"{series_uid}"), exist_ok=True)
            zip_path = os.path.join(dest_folder, 'compressed', f"{series_uid}", f"{series_uid}.zip")
            if os.path.exists(zip_path):
                return  # Skip if already downloaded

            #api_url = "https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage"
            api_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
            if 'NCI_ISBI' in name:
                api_url = "https://public.cancerimagingarchive.net/nbia-api/services/v1/getImage"

            params = {"SeriesInstanceUID": series_uid}

            response = requests.get(api_url, params=params, stream=True, timeout=480)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                tqdm.write(f"[{series_uid}] Failed — HTTP {response.status_code}")
                os.remove(zip_path)
        except Exception as e:
            tqdm.write(f"[{series_uid}] Error: {e}")

    num_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Downloading {len(series_uids)} series with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(download_series, series_uids), total=len(series_uids), desc="Downloading"))

    print("✅ All downloads complete.")


def handle_zenodo(name, url, dest_folder):
    """
    Handle downloading from Zenodo using wget.
    """
    import subprocess
    import os
    
    os.makedirs(dest_folder, exist_ok=True)
    
    if already_downloaded(dest_folder):
        print(f"✅ {name} already exists at {dest_folder}. Skipping download.")
        return
    
    print(f"📥 Downloading {name} from Zenodo → {dest_folder}")
    
    if isinstance(url, str):
        urls = [url]
    else:
        urls = url
    
    for idx, single_url in enumerate(urls):
        if len(urls) > 1:
            output = os.path.join(dest_folder, f"{name}_{idx}.zip")
        else:
            output = os.path.join(dest_folder, f"{name}.zip")
        
        if os.path.exists(output) and os.path.getsize(output) > 0:
            print(f"⏭️  {os.path.basename(output)} already exists, skipping")
            continue
        
        print(f"📥 Downloading {os.path.basename(output)}...")
        
        # Simple wget command - same as terminal
        subprocess.run(["wget", "-O", output, single_url])
        
        if os.path.exists(output) and os.path.getsize(output) > 0:
            print(f"✅ Downloaded {os.path.basename(output)}")
        else:
            print(f"❌ Failed to download {os.path.basename(output)}")
    
    print(f"✅ All Zenodo files for {name} downloaded.")
    

def handle_researchgate(name, url, dest_folder):
    output = os.path.join(dest_folder, f"{name}.zip")

    header_user_agent = "--header=User-Agent: Mozilla/5.0"
    header_referer = "--header=Referer: https://www.researchgate.net/"

    if isinstance(url, str):

        subprocess.run([
            f"wget",
            header_user_agent,
            header_referer,
            url,
            "-O", output
        ])
    elif isinstance(url, list):
        for ind, sub_url in enumerate(url):
            output = os.path.join(dest_folder, f"{name}_{ind}.zip")

            subprocess.run([
                f"wget",
                header_user_agent,
                header_referer,
                sub_url,
                "-O", output
            ])
    else:
        print(f"❌ Error downloading from ResearchGate. Unknown type of url {type(url)}")


def handle_kits19(dest_folder):
    download_file('https://github.com/neheller/kits19/archive/refs/heads/master.zip', os.path.join(dest_folder, 'master.zip'), 'KITS19')
    extract_all_zips(dest_folder, 'KITS19')
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_kits19.py'),"datasets/CT/KITS19/extracted/kits19-master/starter_code/download_kits19.py")
    os.system('python datasets/CT/KITS19/extracted/kits19-master/starter_code/download_kits19.py')
    
def handle_kits23(dest_folder):
    download_file('https://github.com/neheller/kits23/archive/refs/heads/main.zip', os.path.join(dest_folder, 'master.zip'), 'KITS23')
    extract_all_zips(dest_folder, 'KITS23')
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_kits23.py'),"datasets/CT/KITS23/extracted/kits23-main/kits23/download_kits23.py")
    os.system('python datasets/CT/KITS23/extracted/kits23-main/kits23/download_kits23.py')


def get_chrome_version():
    """
    Detect installed Chrome major version (for driver compatibility).
    Falls back to 137 if detection fails.
    """
    try:
        output = subprocess.check_output(
            ["google-chrome", "--version"],
            stderr=subprocess.DEVNULL
        ).decode()

        match = re.search(r"(\d+)\.", output)
        if match:
            return int(match.group(1))
    except:
        pass

    return 137


def handle_kingston(name, url, dest_folder):    
    os.makedirs(dest_folder, exist_ok=True)

    print(f"📥 Downloading {name} via Kingston → {dest_folder}")

    chrome_version = get_chrome_version()
    print(f"🧠 Using Chrome version: {chrome_version}")

    # -------------------------
    # Find Chrome binary
    # -------------------------
    chrome_paths = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
        "/snap/bin/google-chrome",
        "/opt/google/chrome/chrome",
        "/usr/local/bin/chrome",
        "/usr/local/bin/google-chrome",
    ]
    
    chrome_binary = None
    for path in chrome_paths:
        if os.path.exists(path):
            chrome_binary = path
            print(f"✅ Found Chrome at: {chrome_binary}")
            break
    
    if not chrome_binary:
        print("⚠️  Chrome not found. Trying without specifying binary path...")
        print("   This may fail if Chrome is not installed.")

    # -------------------------
    # Chrome setup
    # -------------------------
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless=new")  # Run headless (no GUI)
    
    # Set binary location if found
    if chrome_binary:
        options.binary_location = chrome_binary

    prefs = {
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "plugins.always_open_pdf_externally": True,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    options.add_experimental_option("prefs", prefs)

    # -------------------------
    # Try to create driver with fallback
    # -------------------------
    try:
        print("🔄 Starting Chrome driver...")
        driver = uc.Chrome(
            options=options,
            version_main=chrome_version,
            use_subprocess=True
        )
    except Exception as e:
        print(f"❌ Failed to create Chrome driver: {e}")
        print("💡 Falling back to manual download...")
        print(f"   Please manually download: {url}")
        print(f"   And place files in: {dest_folder}")
        return

    try:
        # Force download directory
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {
                "behavior": "allow",
                "downloadPath": os.path.abspath(dest_folder)
            }
        )

        # Open page
        print("🌐 Opening page...")
        driver.get(url)
        time.sleep(10)  # Wait for download to start

        print(f"✅ {name} download initiated → {dest_folder}")
        print("   Check the folder for the downloaded file.")

    except Exception as e:
        print(f"❌ Kingston download failed: {e}")
        print("💡 Try manually downloading the file.")

    finally:
        try:
            driver.quit()
        except:
            pass


def handle_resect(dest_folder, user_agent=None, verify_ssl=True):
    download_file('https://files.de-1.osf.io/v1/resources/jv8bk/providers/osfstorage/?zip=', os.path.join(dest_folder, 'RESECT_OSF.zip'), 'RESECT')
    base_url = 'https://ns9999k.webs.sigma2.no/10.11582_2017.00004/RESECT/NIFTI'
    for i in range(1, 28):
        if i in [9, 10, 20, 22]:
            continue
        download_file(f'{base_url}/Case{i}/MRI/Case{i}-FLAIR.nii.gz', os.path.join(dest_folder, 'extracted', f'Case{i}-FLAIR.nii.gz'), 'RESECT')
        download_file(f'{base_url}/Case{i}/MRI/Case{i}-T1.nii.gz', os.path.join(dest_folder, 'extracted', f'Case{i}-T1.nii.gz'), 'RESECT')


def get_filename_from_url(url, name, source):
    if isinstance(url, list):
        return [os.path.basename(urlparse(u).path) for u in url]
    if source == "mendeley":
        return f"{name}.zip"
    return os.path.basename(urlparse(url).path)


# ============================================================================
# BASE CLASSES AND MIXINS
# ============================================================================

class SourceHandler:
    """Base class for all source handlers."""
    
    def __init__(self, name, url, dest_folder, **kwargs):
        self.name = name
        self.url = url
        self.dest_folder = dest_folder
        self.kwargs = kwargs
        
    def can_handle(self, source):
        """Check if this handler can process the given source."""
        return False
    
    def process(self):
        """Process the download for this source."""
        raise NotImplementedError
    
    def _extract_urls(self, domains):
        """Extract URLs matching the given domains."""
        if isinstance(self.url, list):
            return [el for el in self.url if any(d in el for d in domains)]
        return self.url
    
    def _handle_other_urls(self, domains):
        """Handle any URLs that don't match the given domains."""
        if isinstance(self.url, list):
            other = [el for el in self.url if not any(d in el for d in domains)]
            if other:
                return download_dataset(self.name, other, **self.kwargs)
        return None
    
    def _is_already_downloaded(self):
        """Check if the dataset is already downloaded."""
        return already_downloaded(self.dest_folder)


class MultiPartMixin:
    """Mixin for handling multi-part downloads."""
    
    def _get_output_path(self, index=None):
        """Get the output path for a file."""
        if index is not None and len(self.url) > 1:
            return os.path.join(self.dest_folder, f"{self.name}_{index}.zip")
        return os.path.join(self.dest_folder, f"{self.name}.zip")


# ============================================================================
# SOURCE HANDLERS
# ============================================================================

class FigshareHandler(SourceHandler):
    """Handler for Figshare datasets."""
    
    def can_handle(self, source):
        return source == "figshare"
    
    def process(self):
        os.makedirs(self.dest_folder, exist_ok=True)
        
        if self._is_already_downloaded():
            print(f"✅ {self.name} already exists at {self.dest_folder}. Skipping download.")
            return
        
        print(f"📥 Downloading {self.name} from Figshare → {self.dest_folder}")
        
        urls = self.url if isinstance(self.url, list) else [self.url]
        
        for idx, single_url in enumerate(urls):
            article_id = single_url.strip().split('/')[-1]
            self._download_figshare_article(article_id, idx, len(urls))
        
        # Handle any non-Figshare URLs
        non_figshare_urls = [el for el in urls if "figshare" not in el and "rdr" not in el]
        if non_figshare_urls:
            download_dataset(self.name, non_figshare_urls, **self.kwargs)
        
        print(f"✅ All Figshare files for {self.name} downloaded.")
    
    def _download_figshare_article(self, article_id, idx, total):
        """Download a single Figshare article."""
        print(f"🔍 Fetching file info from Figshare API for article {article_id}...")
        r = requests.get(f'{FIGSHARE_API_URL}/articles/{article_id}/files')
        
        if r.status_code != 200:
            print(f'❌ Failed to get file info: {r.content}')
            return
        
        file_metadata = r.json()
        
        for file_info in file_metadata:
            file_id = file_info['id']
            file_name = file_info['name']
            
            dest_path = self._get_file_path(file_name, len(file_metadata) > 1, idx)
            
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                print(f"⏭️  {file_name} already exists, skipping")
                continue
            
            self._download_file(file_id, file_name, dest_path)
    
    def _get_file_path(self, file_name, multiple_files, idx):
        """Determine the destination path for a file."""
        if multiple_files:
            return os.path.join(self.dest_folder, file_name)
        return os.path.join(self.dest_folder, f"{self.name}_{idx}.zip")
    
    def _download_file(self, file_id, file_name, dest_path):
        """Download a file from Figshare."""
        print(f"📥 Downloading {file_name}...")
        response = requests.get(
            f'{FIGSHARE_API_URL}/file/download/{file_id}',
            stream=True
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to download {file_name}: {response.status_code}")
            return
        
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name) as pbar:
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Downloaded {file_name}")


class ZenodoHandler(SourceHandler):
    """Handler for Zenodo datasets."""
    
    def can_handle(self, source):
        return source == "zenodo"
    
    def process(self):
        os.makedirs(self.dest_folder, exist_ok=True)
        
        if self._is_already_downloaded():
            print(f"✅ {self.name} already exists at {self.dest_folder}. Skipping download.")
            return
        
        print(f"📥 Downloading {self.name} from Zenodo → {self.dest_folder}")
        
        urls = self.url if isinstance(self.url, list) else [self.url]
        
        for idx, single_url in enumerate(urls):
            output = self._get_output_path(idx, len(urls))
            
            if os.path.exists(output) and os.path.getsize(output) > 0:
                print(f"⏭️  {os.path.basename(output)} already exists, skipping")
                continue
            
            print(f"📥 Downloading {os.path.basename(output)}...")
            subprocess.run(["wget", "-O", output, single_url])
            
            if os.path.exists(output) and os.path.getsize(output) > 0:
                print(f"✅ Downloaded {os.path.basename(output)}")
            else:
                print(f"❌ Failed to download {os.path.basename(output)}")
        
        # Handle any non-Zenodo URLs
        if isinstance(self.url, list):
            non_zenodo_urls = [el for el in self.url if "zenodo" not in el]
            if non_zenodo_urls:
                download_dataset(self.name, non_zenodo_urls, **self.kwargs)
        
        print(f"✅ All Zenodo files for {self.name} downloaded.")
    
    def _get_output_path(self, idx, total):
        """Get the output path for a file."""
        if total > 1:
            return os.path.join(self.dest_folder, f"{self.name}_{idx}.zip")
        return os.path.join(self.dest_folder, f"{self.name}.zip")


class DataverseHandler(SourceHandler):
    """Handler for Dataverse datasets."""
    
    def can_handle(self, source):
        return source == "dataverse"
    
    def process(self):
        os.makedirs(self.dest_folder, exist_ok=True)
        
        if self._is_already_downloaded():
            print(f"✅ {self.name} already exists at {self.dest_folder}. Skipping download.")
            return
        
        print(f"📥 Downloading {self.name} from Dataverse → {self.dest_folder}")
        
        urls = self.url if isinstance(self.url, list) else [self.url]
        
        def download_dataverse_file(file_url):
            try:
                filename = os.path.basename(urlparse(file_url).path)
                if not filename:
                    filename = f"{self.name}.zip"
                
                dest_path = os.path.join(self.dest_folder, filename)
                if os.path.exists(dest_path):
                    tqdm.write(f"⏭️ Skipping {filename}, already exists.")
                    return
                
                with requests.get(file_url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("content-length", 0))
                    pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=filename)
                    
                    with open(dest_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                    pbar.close()
                tqdm.write(f"✅ Downloaded {filename}")
            except Exception as e:
                tqdm.write(f"❌ Error downloading {file_url}: {e}")
        
        num_workers = min(multiprocessing.cpu_count(), 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(download_dataverse_file, urls), total=len(urls), desc="Files"))
        
        print(f"✅ All Dataverse files for {self.name} downloaded.")


class TCIASeriesHandler(SourceHandler):
    """Handler for TCIA datasets (series-based)."""
    
    def can_handle(self, source):
        return source == "TCIA"
    
    def process(self):
        os.makedirs(self.dest_folder, exist_ok=True)
        print(f"📥  Downloading {self.name}...")
        
        # Handle special case for HCC_TACE_SEG
        if self.name == 'HCC_TACE_SEG':
            excel_path = os.path.join(self.dest_folder, 'HCC-TACE-Seg_v1_202201-nbia-digest.xlsx')
            if not os.path.exists(excel_path):
                download_file(
                    'https://www.cancerimagingarchive.net/wp-content/uploads/HCC-TACE-Seg_v1_202201-nbia-digest.xlsx',
                    excel_path,
                    self.name
                )
        
        # Get series UIDs
        series_uids = self._get_series_uids()
        
        # Download all series
        self._download_series(series_uids)
        
        print("✅ All downloads complete.")
    
    def _get_series_uids(self):
        """Get series UIDs from .tcia manifest or collection name."""
        if isinstance(self.url, str) and self.url.endswith('.tcia'):
            return self._get_uids_from_manifest()
        else:
            return self._get_uids_from_collection()
    
    def _get_uids_from_manifest(self):
        """Get UIDs from a .tcia manifest file."""
        if 'datasets' in self.url:
            file_path = self.url
        else:
            file_name = self.url.split('/')[-1]
            file_path = os.path.join(self.dest_folder, file_name)
            
            print(f"Detected .tcia manifest file. Downloading {file_name} ...")
            response = requests.get(self.url, timeout=30)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Manifest downloaded to {file_path}")
            else:
                raise Exception(f"Failed to download manifest: HTTP {response.status_code}")
        
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if '1.' in line]
    
    def _get_uids_from_collection(self):
        """Get UIDs from a TCIA collection name."""
        collection_name = self.url.rstrip('/').split('/')[-1]
        if 'ct-colonography' in collection_name:
            collection_name = 'ct colonography'
        
        print(f"Fetching series list from collection: {collection_name} ...")
        series_list = nbia.getSeries(collection_name)
        return [s['SeriesInstanceUID'] for s in series_list]
    
    def _download_series(self, series_uids):
        """Download multiple series with multithreading."""
        def download_series(series_uid):
            try:
                os.makedirs(os.path.join(self.dest_folder, 'compressed', f"{series_uid}"), exist_ok=True)
                zip_path = os.path.join(self.dest_folder, 'compressed', f"{series_uid}", f"{series_uid}.zip")
                
                if os.path.exists(zip_path):
                    return
                
                api_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
                if 'NCI_ISBI' in self.name:
                    api_url = "https://public.cancerimagingarchive.net/nbia-api/services/v1/getImage"
                
                params = {"SeriesInstanceUID": series_uid}
                response = requests.get(api_url, params=params, stream=True, timeout=480)
                
                if response.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    tqdm.write(f"[{series_uid}] Failed — HTTP {response.status_code}")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
            except Exception as e:
                tqdm.write(f"[{series_uid}] Error: {e}")
        
        num_workers = min(multiprocessing.cpu_count(), 8)
        print(f"Downloading {len(series_uids)} series with {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(download_series, series_uids), total=len(series_uids), desc="Downloading"))


class GDriveHandler(SourceHandler):
    """Handler for Google Drive datasets."""
    
    def can_handle(self, source):
        return source == "gdown"
    
    def process(self):
        if self._is_already_downloaded():
            print(f"✅ {self.name} already exists at {self.dest_folder}. Skipping download.")
            return
        
        urls = self.url if isinstance(self.url, list) else [self.url]
        for url_ in urls:
            download_gdown(url_, self.dest_folder, self.name)


class KingstonHandler(SourceHandler):
    """Handler for Kingston (browser-based) downloads."""
    
    def can_handle(self, source):
        return source == "kingston"
    
    def process(self):
        if self._is_already_downloaded():
            print(f"✅ {self.name} already exists at {self.dest_folder}. Skipping download.")
            return
        
        os.makedirs(self.dest_folder, exist_ok=True)
        print(f"📥 Downloading {self.name} via Kingston → {self.dest_folder}")
        
        chrome_version = get_chrome_version()
        print(f"🧠 Using Chrome version: {chrome_version}")
        
        chrome_binary = self._find_chrome_binary()
        driver = self._setup_chrome_driver(chrome_binary, chrome_version)
        
        if not driver:
            print(f"💡 Please manually download: {self.url}")
            print(f"   And place files in: {self.dest_folder}")
            return
        
        try:
            driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": os.path.abspath(self.dest_folder)}
            )
            
            print("🌐 Opening page...")
            driver.get(self.url)
            time.sleep(10)
            
            print(f"✅ {self.name} download initiated → {self.dest_folder}")
        except Exception as e:
            print(f"❌ Kingston download failed: {e}")
            print("💡 Try manually downloading the file.")
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def _find_chrome_binary(self):
        """Find the Chrome binary path."""
        chrome_paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/snap/bin/chromium",
            "/snap/bin/google-chrome",
            "/opt/google/chrome/chrome",
            "/usr/local/bin/chrome",
            "/usr/local/bin/google-chrome",
        ]
        
        for path in chrome_paths:
            if os.path.exists(path):
                print(f"✅ Found Chrome at: {path}")
                return path
        
        print("⚠️  Chrome not found. Trying without specifying binary path...")
        return None
    
    def _setup_chrome_driver(self, chrome_binary, chrome_version):
        """Set up the Chrome driver."""
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--headless=new")
        
        if chrome_binary:
            options.binary_location = chrome_binary
        
        prefs = {
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_setting_values.automatic_downloads": 1,
        }
        options.add_experimental_option("prefs", prefs)
        
        try:
            print("🔄 Starting Chrome driver...")
            return uc.Chrome(options=options, version_main=chrome_version, use_subprocess=True)
        except Exception as e:
            print(f"❌ Failed to create Chrome driver: {e}")
            return None


class ResearchGateHandler(SourceHandler):
    """Handler for ResearchGate datasets."""
    
    def can_handle(self, source):
        return source == "researchgate"
    
    def process(self):
        header_user_agent = "--header=User-Agent: Mozilla/5.0"
        header_referer = "--header=Referer: https://www.researchgate.net/"
        
        urls = self.url if isinstance(self.url, list) else [self.url]
        
        for ind, sub_url in enumerate(urls):
            output = os.path.join(self.dest_folder, f"{self.name}_{ind}.zip") if len(urls) > 1 else os.path.join(self.dest_folder, f"{self.name}.zip")
            
            subprocess.run([
                "wget",
                header_user_agent,
                header_referer,
                sub_url,
                "-O", output
            ])


class GithubHandler(SourceHandler):
    """Handler for GitHub datasets."""
    
    def can_handle(self, source):
        return source == "github"
    
    def process(self):
        # Extract subdirectory path from URL if needed
        if isinstance(self.url, str):
            repo_url = self.url
            subdir_path = None
            
            if '/tree/' in self.url:
                parts = self.url.split('/tree/')
                repo_url = parts[0]
                subdir_path = parts[1].split('/')[-1]  # Get last part as subdir
            else:
                # For full repo download, handle separately
                pass
            
            if subdir_path:
                download_github_subdir(repo_url, subdir_path, self.dest_folder)
            else:
                # Download entire repo
                download_github_repo(repo_url, self.dest_folder)
        elif isinstance(self.url, list):
            for url_ in self.url:
                # Handle each URL
                pass


class MendeleyHandler(SourceHandler):
    """Handler for Mendeley datasets."""
    
    def can_handle(self, source):
        return source == "mendeley"
    
    def process(self):
        user_agent = "Mozilla/5.0"
        self._download_with_headers(user_agent)
    
    def _download_with_headers(self, user_agent):
        """Download with custom headers."""
        urls = self.url if isinstance(self.url, list) else [self.url]
        
        for idx, url_ in enumerate(urls):
            filename = f"{self.name}.zip" if len(urls) == 1 else f"{self.name}_{idx}.zip"
            dest_path = os.path.join(self.dest_folder, filename)
            download_file(url_, dest_path, self.name, user_agent=user_agent)


# ============================================================================
# DATASET REGISTRY
# ============================================================================

class DatasetRegistry:
    """Registry for special dataset handlers."""
    
    _handlers = {
        'KITS19': handle_kits19,
        'KITS23': handle_kits23,
        'RESECT': handle_resect,
        'INTRARETINAL_CYSTOID_FLUID': '_handle_intraretinal',
        'NEVUS': '_handle_nevus',
        'TEETH_SEG': '_handle_teeth_seg',
        'BCN20000': '_handle_bcn20000',
        'ORVS': '_handle_orvs',
    }
    
    @classmethod
    def handle_special(cls, name, url, dest_folder, **kwargs):
        """Handle special datasets that need custom logic."""
        handler = cls._handlers.get(name)
        if handler:
            if callable(handler):
                handler(dest_folder)
            else:
                # String method name for dataset_tools imports
                getattr(cls, handler)(dest_folder)
            return True
        return False
    
    @staticmethod
    def _handle_intraretinal(dest_folder):
        import dataset_tools as dtools
        dtools.download(dataset='Intraretinal Cystoid Fluid', dst_dir=dest_folder)
    
    @staticmethod
    def _handle_nevus(dest_folder):
        import dataset_tools as dtools
        dtools.download(dataset='Accurate Nevus Shapes', dst_dir=dest_folder)
    
    @staticmethod
    def _handle_teeth_seg(dest_folder):
        import dataset_tools as dtools
        dtools.download(dataset='Teeth Segmentation on Dental X-ray Images', dst_dir=dest_folder)
    
    @staticmethod
    def _handle_bcn20000(dest_folder):
        os.system('isic image download --collections 249 datasets/Dermoscopy/BCN20000')
    
    @staticmethod
    def _handle_orvs(dest_folder):
        if 'ORVS' in dest_folder:
            download_github_subdir(
                repo_url='https://github.com/AbdullahSarhan/ICPRVessels',
                subdir_path='Vessels-Datasets/ORVS',
                dest_dir=os.path.dirname(dest_folder)
            )


# ============================================================================
# SOURCE DISPATCHER
# ============================================================================

class SourceDispatcher:
    """Main dispatcher for routing to appropriate source handlers."""
    
    _handlers = [
        FigshareHandler,
        ZenodoHandler,
        DataverseHandler,
        TCIASeriesHandler,
        GDriveHandler,
        KingstonHandler,
        ResearchGateHandler,
        GithubHandler,
        MendeleyHandler,
    ]
    
    @classmethod
    def dispatch(cls, name, url, modality, base_dir="datasets", filter_pattern=None, new_folder=False):
        """Dispatch to the appropriate handler."""
        dest_folder = os.path.join(base_dir, modality, name)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Check for special datasets first
        if DatasetRegistry.handle_special(name, url, dest_folder, modality=modality, base_dir=base_dir):
            return
        
        # Handle TCIA manifest cases
        if isinstance(url, list) and any(".tcia" in el for el in url):
            for el in url:
                cls.dispatch(name, el, modality, base_dir, filter_pattern, new_folder)
            return
        
        # Get source type
        source = get_source(url)
        
        # Find and execute handler
        for handler_class in cls._handlers:
            handler = handler_class(
                name, url, dest_folder,
                modality=modality,
                base_dir=base_dir,
                filter_pattern=filter_pattern,
                new_folder=new_folder
            )
            if handler.can_handle(source):
                handler.process()
                break
        else:
            # Fallback to generic download
            cls._generic_download(name, url, dest_folder, source, filter_pattern, new_folder)
    
    @staticmethod
    def _generic_download(name, url, dest_folder, source, filter_pattern, new_folder):
        """Generic download for unknown sources."""
        user_agent = "Mozilla/5.0" if source == "mendeley" else None
        verify_ssl = False if (isinstance(url, str) and ('weiss-develop' in url or 'isbi14' in url or name == 'SPINE' or name == 'BUSI' or 'KVASIR' in name)) else True
        
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        urls = url if isinstance(url, list) else [url]
        filenames = get_filename_from_url(url, name, source)
        
        for idx, (u, filename) in enumerate(zip(urls, filenames if isinstance(filenames, list) else [filenames])):
            dest_path = os.path.join(dest_folder, filename)
            part_msg = f' part {idx + 1} of {len(urls)}' if len(urls) > 1 else ''
            
            if already_downloaded(dest_path):
                print(f"✅ {name} already exists at {dest_path}. Skipping download.")
                continue
            
            if not download_file(u, dest_path, name, user_agent=user_agent, verify_ssl=verify_ssl, part=part_msg):
                return
        
        # Extract archives
        if os.path.exists(os.path.join(dest_folder, 'converted')):
            print(f'⚠️  The dataset {name} is already converted.')
        else:
            extract_all_zips(dest_folder, name, filter_pattern=filter_pattern, new_folder=new_folder)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def download_dataset(name, url, modality, base_dir="datasets", filter_pattern=None, new_folder=False):
    """
    Main entry point for downloading datasets.
    
    Args:
        name (str): Name of the dataset
        url (str or list): URL(s) to download from
        modality (str): Modality type (e.g., 'CT', 'MRI', 'Dermoscopy')
        base_dir (str): Base directory for datasets
        filter_pattern (str, optional): Pattern to filter files during extraction
        new_folder (bool): Whether to create a new folder for each archive
    """
    SourceDispatcher.dispatch(name, url, modality, base_dir, filter_pattern, new_folder)


if __name__ == "__main__":
    pass
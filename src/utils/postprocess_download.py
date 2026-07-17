import os
import zipfile
from tqdm import tqdm
import rarfile
import tarfile
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shutil

def is_compressed_file(file_path):
    """Check if a file is a valid ZIP, RAR, or TAR archive (even without a proper extension)."""
    if not os.path.isfile(file_path):
        return False
    if 'ULS23' in file_path:
        return '.zip' in file_path


    return (
        zipfile.is_zipfile(file_path) or
        rarfile.is_rarfile(file_path) or
        is_tarfile_safe(file_path) or
        '.part' in file_path or
        '.7z' in file_path or 
        ('WAW_TACE' in file_path and 'files-archive' in file_path)
    )

def is_tarfile_safe(file_path):
    """Safely check if a file is a tar archive (supports .tar, .tar.gz, .tgz, etc)."""
    try:
        return tarfile.is_tarfile(file_path)
    except Exception:
        return False


def is_unzipped(extract_to, zip_path, filter_pattern=None):
    """
    Check if the compressed file (zip/rar/tar) has already been extracted
    by verifying if the expected content exists in the extraction folder.
    """

    if not is_compressed_file(zip_path):
        return False
    if '.7z' in zip_path:
        return False

    # Get expected files from inside the archive
    try:
        if zipfile.is_zipfile(zip_path) or ('.part' in zip_path and 'OCT5K' not in zip_path and 'YEAZ' not in zip_path and 'MSD' not in zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                expected_files = zip_ref.namelist()
        elif rarfile.is_rarfile(zip_path):
            with rarfile.RarFile(zip_path, 'r') as rar_ref:
                expected_files = rar_ref.namelist()
        elif tarfile.is_tarfile(zip_path):
            with tarfile.open(zip_path, 'r:*') as tar_ref:
                expected_files = tar_ref.getnames()
        else:
            raise ValueError(f"[❌ ERROR] File '{zip_path}' is not a valid zip, rar, or tar file!")
    except Exception as e:
        if 'WAW_TACE' not in zip_path:
            print(f"[❌ ERROR] Failed to read archive contents: {e}")
        return False

    # Check if expected files exist in extract_to
    if filter_pattern is not None:
        new_members = []
        for expected_file in expected_files:
            for pattern in filter_pattern:
                if expected_file.startswith(pattern):
                    new_members.append(expected_file)
        expected_files = new_members
    for expected_file in expected_files:
        expected_path = os.path.join(extract_to, expected_file)
        if not os.path.exists(expected_path):
            return False

    return True

def extract_member(rar_ref, member, extract_to):
    """Extract a single member of the RAR archive."""
    if 'OCT5K' in extract_to:
        rar_ref.extract(member, path=extract_to, pwd='MCME2017'.encode())
    else:
        rar_ref.extract(member, path=extract_to)

def extract_with_7z(zip_path, output_dir, password=None):
    print('Extracting with 7z... (this might take a while)')
    os.makedirs(output_dir, exist_ok=True)
    command = '7z' if shutil.which("7z") else '7zz'
    

    
    if password is not None:
        result = subprocess.run(
            [command, 'x', '-y', f'-p{password}', '-o' + output_dir, zip_path],
            capture_output=True,
            text=True
        )
    else:
        cmd_args = [command, 'x', '-y', '-o' + output_dir, zip_path]
        if 'STS_TOOTH_CT' in zip_path:
            cmd_args.append('-xr!STS-2D-Tooth')
            cmd_args.append('-xr!Unlabeled')
            cmd_args.append('-xr!ROI')
        if 'STS_TOOTH_XRAY' in zip_path:
            cmd_args.append('-xr!STS-3D-Tooth')
            cmd_args.append('-xr!Unlabeled')
        if 'ULS' in zip_path:
            cmd_args.append('-xr!categories.zip')
            cmd_args.append('-xr!partially_annotated')

        result = subprocess.run(
            cmd_args,
            capture_output=True, 
            text=True
        )

    if result.returncode != 0:
        print("7z extraction failed:")
        print(result.stderr)

    else:
        print("Extraction successful:")
        print(result.stdout)

def unzip_file(zip_path, extract_to=None, filter_pattern=None, verbose=True):

    """Unzip or unrar a file, depending on its format (ZIP or RAR)."""
    if not is_compressed_file(zip_path) or '.DS_Store' in zip_path:
        print(f"❌ Not a zip/rar/tar file: {zip_path}")
        return
    max_workers = 1

    if 'WORD' in zip_path:
        extract_with_7z(zip_path, extract_to, 'word@uestc')
        return
    if 'SPINE/' in zip_path or 'STS_TOOTH_XRAY/files-archive' in zip_path or 'CURVAS' in zip_path or 'STACOM_SLAWT_CT' in zip_path or ('WAW_TACE' in zip_path and '.zip' not in zip_path) or '.7z' in zip_path or ('/ULS/' in zip_path and '.zip' in zip_path and 'labels' not in zip_path):
        extract_with_7z(zip_path, extract_to)
        return
    if 'STS_TOOTH' in zip_path and 'file-archive' not in zip_path:
        if not os.path.exists(zip_path.replace(os.path.basename(zip_path), 'SD-Tooth')):
            extract_with_7z(zip_path.replace(os.path.basename(zip_path), 'SD-Tooth.zip.001'), extract_to)
        return

    if zipfile.is_zipfile(zip_path) or 'TG3K' in zip_path or 'TN3K' in zip_path:
        if extract_to is None:
            extract_to = os.path.splitext(zip_path)[0]
        if verbose:
            print(f"📦 Extracting {zip_path} → {extract_to}")
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            if filter_pattern is not None:
                new_members = []
                for member in members:
                    for pattern in filter_pattern:
                        if member.filename.startswith(pattern):
                            new_members.append(member)
                members = new_members

            with tqdm(total=len(members), desc=f"Unzipping {os.path.basename(zip_path)}", unit='file') as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for member in members:
                        futures.append(executor.submit(extract_member, zip_ref, member, extract_to))

                    # Wait for all the extraction tasks to complete
                    for future in futures:
                        future.result()  # Blocking until each future is done
                        pbar.update(1)
        if verbose:
            print(f"✅ Extracted ZIP to {extract_to}")
    elif rarfile.is_rarfile(zip_path):
        # If it's a RAR file, extract using parallel threads
        if extract_to is None:
            extract_to = os.path.splitext(zip_path)[0]
        if verbose:
            print(f"📦 Extracting {zip_path} → {extract_to}")
        os.makedirs(extract_to, exist_ok=True)

        try:
            with rarfile.RarFile(zip_path, 'r') as rar_ref:
                members = rar_ref.namelist()
                with tqdm(total=len(members), desc=f"Unraring {os.path.basename(zip_path)}", unit='file') as pbar:
                    # Use ThreadPoolExecutor for parallel extraction
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        futures = []
                        for member in members:
                            futures.append(executor.submit(extract_member, rar_ref, member, extract_to))

                        # Wait for all the extraction tasks to complete
                        for future in futures:
                            future.result()  # Blocking until each future is done
                            pbar.update(1)
            if verbose:
                print(f"✅ Extracted RAR to {extract_to}")
        except rarfile.Error as e:
            print(f"❌ Error reading RAR file {zip_path}: {e}")
    elif tarfile.is_tarfile(zip_path) or 'YEAZ' in zip_path:
                if verbose:
                    print(f"📦 Extracting TAR: {zip_path} → {extract_to}")
                with tarfile.open(zip_path, 'r:*') as tar_ref:
                    members = tar_ref.getmembers()
                    with tqdm(total=len(members), desc=f"Untarring {os.path.basename(zip_path)}", unit='file') as pbar:
                        for member in members:
                            extract_member(tar_ref, member, extract_to)
                            pbar.update(1)
                if verbose:
                    print(f"✅ Extracted TAR to {extract_to}")

    else:
        raise ValueError(f"Unsupported archive format: {zip_path}")

if __name__ == "__main__":
    pass

import os
import shutil
from src.utils.dataset_utils import *
from src.converters.utils_ct import *
from tqdm import tqdm
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

def convert_aero_path():
    converted_path = os.path.join('datasets', 'CT', 'AERO_PATH', 'converted')
    label_classes = ['airways', 'lungs']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'AERO_PATH', 'extracted', 'AeroPath')
    image_paths = [os.path.join(extracted_base, f'{i}', f'{i}_CT_HR.nii.gz') for i in np.arange(1, 28)]

    filename_map = {}
    lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        list(tqdm(
            executor.map(lambda args: process_aero_file(args, converted_path, filename_map, lock),
                         enumerate(image_paths)),
            total=len(image_paths),
            desc="🖼️  Processing AERO_PATH images"
        ))

    save_mapping(converted_path, filename_map)

def process_aero_file(idx_file_pair, converted_path, filename_map, lock):
    idx, file = idx_file_pair
    filename = f"{idx:06d}.nii.gz"

    dest_image = os.path.join(converted_path, 'images', filename)
    dest_lungs = os.path.join(converted_path, 'labels', 'lungs', filename)
    dest_airways = os.path.join(converted_path, 'labels', 'airways', filename)

    shutil.copy(file, dest_image)
    shutil.copy(file.replace('.nii.gz', '_label_lungs.nii.gz'), dest_lungs)
    shutil.copy(file.replace('.nii.gz', '_label_airways.nii.gz'), dest_airways)

    with lock:
        filename_map[filename] = {
            'image': file,
            'lungs': file.replace('.nii.gz', '_label_lungs.nii.gz'),
            'airways': file.replace('.nii.gz', '_label_airways.nii.gz')
        }


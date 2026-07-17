import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.utils.dataset_utils import *
from src.converters.utils_ct import *


def convert_luna():
    converted_path = os.path.join('datasets', 'CT', 'LUNA', 'converted')
    label_classes = ['left_lung', 'right_lung', 'airways', 'lungs']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'LUNA', 'extracted')
    for i in np.arange(0, 2):
        for zip_file in os.listdir(os.path.join(extracted_base, f'files-archive_{i}')):
            if '.zip' not in zip_file:
                continue
            if not is_unzipped(os.path.join(extracted_base, f'files-archive_{i}'), os.path.join(extracted_base, f'files-archive_{i}', zip_file)):
                unzip_file(os.path.join(extracted_base, f'files-archive_{i}', zip_file), extract_to=os.path.join(extracted_base, f'files-archive_{i}'))
    image_paths = []
    for i in range(0, 7):
        subset_path = os.path.join(extracted_base, 'files-archive_0', f'subset{i}')
        image_paths += [os.path.join(subset_path, f) for f in os.listdir(subset_path) if f.endswith('.mhd')]
    for i in range(7, 10):
        subset_path = os.path.join(extracted_base, 'files-archive_1', f'subset{i}')
        image_paths += [os.path.join(subset_path, f) for f in os.listdir(subset_path) if f.endswith('.mhd')]


    args_list = [(file, idx, converted_path, label_classes) for idx, file in enumerate(image_paths)]

    filename_map = {}
    with ProcessPoolExecutor() as executor:
        for filename, data in tqdm(executor.map(process_luna_case, args_list), total=len(args_list), desc="🖼️  Processing LUNA images"):
            filename_map[filename] = data

    save_mapping(converted_path, filename_map)

def process_luna_case(args):
    file, idx, converted_path, label_classes = args
    label_indices = [3, 4, 5]

    filename = f"{idx:06d}.nii.gz"
    convert_mhd_raw_to_nifti(file, os.path.join(converted_path, 'images', filename))
    label_file = file.replace('files-archive_1', 'files-archive_0')
    for i in range(0, 10):
        label_file = label_file.replace(f'subset{i}', 'seg-lungs-LUNA16')


    convert_mhd_raw_to_nifti(label_file, os.path.join(converted_path, 'labels', 'temp', filename), label_classes=label_classes, label_indices=label_indices)

    return filename, {
        'image': [file, file.replace('mhd', 'raw')],
        **{cls: [label_file, label_file.replace('mhd', 'zraw')] 
           for cls in ['left_lung', 'right_lung', 'airways', 'lungs']}
    }

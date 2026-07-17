import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.dataset_utils import *
from src.converters.utils_ct import *

def convert_cdemris():
    converted_path = os.path.join('datasets', 'MRI', 'CDEMRIS', 'converted')

    # Create label mapping
    label_classes = ['left_atrium']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'CDEMRIS', 'extracted')

    image_paths = []
    for i in range(1, 31):
        image_paths += [os.path.join(extracted_base, 'pre', f'p{i}', f'de_a_{i}.nrrd')]
        image_paths += [os.path.join(extracted_base, 'post', f'p{i}', f'de_b_{i}.nrrd')]


    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        
        nii_img = convert_nrrd_nii(file)
        nib.save(nii_img, dest_image_path)

        label_file = file.replace('de_', 'la_seg_')
        nii_img = convert_nrrd_nii(label_file)
        nib.save(nii_img, os.path.join(converted_path, 'labels', 'left_atrium', filename))

        return filename, {'image': file, 'left_atrium': label_file}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing CDEMRIS images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

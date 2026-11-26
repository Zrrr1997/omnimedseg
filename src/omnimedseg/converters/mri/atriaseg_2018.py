from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
from collections import defaultdict
import nibabel as nib

def convert_atriaseg_2018():
    converted_path = os.path.join(data_root(), 'MRI', 'ATRIASEG_2018', 'converted')

    # Create label mapping
    label_classes = ['left_atrium_cavity', 'left_atrium_wall']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'ATRIASEG_2018', 'extracted')

    image_paths = []
    for split in ['Training Set', 'Testing Set']:
        patients = [os.path.join(extracted_base, split, f) for f in os.listdir(os.path.join(extracted_base, split))]
        for patient in patients:
            image_paths += [os.path.join(patient, el) for el in os.listdir(patient) if 'lgemri' in el]

    filename_map = {}
    split_map = defaultdict(list)


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"
        dest_image_path = os.path.join(converted_path, 'images', filename)
        nii_img = convert_nrrd_nii(file)
        nib.save(nii_img, dest_image_path)
        temp_dict = {'image': file}

        label_file_cavity = file.replace('lgemri', 'laendo')
        nii_img = convert_nrrd_nii(label_file_cavity)
        nib.save(nii_img, os.path.join(converted_path, 'labels', 'left_atrium_cavity', filename))
        temp_dict['left_atrium_cavity'] = label_file_cavity
        
        label_file_wall = file.replace('lgemri', 'lawall')
        nii_img = convert_nrrd_nii(label_file_wall)
        nib.save(nii_img, os.path.join(converted_path, 'labels', 'left_atrium_wall', filename))
        temp_dict['left_atrium_wall'] = label_file_wall


        if 'Training' in file:
            split = 'train'
        elif 'Testing' in file:
            split = 'test'


        return filename, temp_dict, split

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing ATRIASEG_2018 images"):
            filename, temp_dict, split = f.result()
            filename_map[filename] = temp_dict
            split_map[split].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, dict(split_map), json_fn='split_map')
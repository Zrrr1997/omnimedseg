from omnimedseg.core.paths import data_root
import os
import shutil
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import nibabel as nib


def convert_pddca():
    converted_path = os.path.join(data_root(), 'CT', 'PDDCA', 'converted')
    label_classes = ['brain_stem', 'chiasm', 'mandible', 'optic_nerve_left', 'optic_nerve_right', 'parotid_left', 'parotid_right', 'submandibular_left', 'submandibular_right']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'PDDCA', 'extracted')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if '0522' in f]

    label_map = {
        "BrainStem": 'brain_stem',
        'Chiasm': 'chiasm',
        'Mandible': 'mandible',
        'OpticNerve_L': 'optic_nerve_left',
        'OpticNerve_R': 'optic_nerve_right',
        'Parotid_L': 'parotid_left',
        'Parotid_R': 'parotid_right',
        'Submandibular_L': 'submandibular_left',
        'Submandibular_R': 'submandibular_right'
    }

    filename_map = {}

    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, path in enumerate(image_paths):
            futures.append(executor.submit(
                convert_pddca_case, idx, path, converted_path, label_classes, label_map
            ))

        for f in tqdm(futures, desc="Processing PDDCA images"):
            filename, case_map = f.result()
            filename_map[filename] = case_map

    save_mapping(converted_path, filename_map)

def convert_pddca_case(idx, path, converted_path, label_classes, label_map):
    filename = f"{idx:06d}.nii.gz"
    image_file = os.path.join(path, 'img.nrrd')
    image = convert_nrrd_nii(image_file)
    nib.save(image, os.path.join(converted_path, 'images', filename))

    label_dir = os.path.join(path, 'structures')
    case_map = {'image': image_file}

    for label_filename in os.listdir(label_dir):
        label_name = os.path.splitext(label_filename)[0]
        label_key = label_map.get(label_name)
        if label_key:
            label_path = os.path.join(label_dir, label_filename)
            label_img = convert_nrrd_nii(label_path)
            nib.save(label_img, os.path.join(converted_path, 'labels', label_key, filename))
            case_map[label_key] = label_path

    return filename, case_map

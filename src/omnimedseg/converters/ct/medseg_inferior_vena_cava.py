import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
import SimpleITK as sitk
import numpy as np


def _process_medseg_inferior_vena_cava(args):
    idx, file, converted_path, label_classes = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)
    data = {'image': file}

    dst_label = os.path.join(converted_path, 'labels', 'inferior_vena_cava', filename)
    label_file = file.replace('30508407', '30508368')
    shutil.copy(label_file, dst_label)
    data['inferior_vena_cava'] = label_file

    return filename, data


def convert_medseg_inferior_vena_cava(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'MEDSEG_INFERIOR_VENA_CAVA', 'converted')
    label_classes = ['inferior_vena_cava']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'MEDSEG_INFERIOR_VENA_CAVA', 'extracted')
    
    image_paths = [os.path.join(extracted_base, '30508407', el) for el in os.listdir(os.path.join(extracted_base, '30508407')) if el.endswith('.nii.gz')]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_medseg_inferior_vena_cava, (i, file, converted_path, label_classes)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing MEDSEG_INFERIOR_VENA_CAVA images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

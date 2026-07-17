import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import SimpleITK as sitk
from src.utils.dataset_utils import *
from src.converters.utils_ct import *

def convert_medseg_ventricles():
    converted_path = os.path.join('datasets', 'MRI', 'MEDSEG_VENTRICLES', 'converted')

    # Create label mapping
    label_classes = ['ventricles']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'MEDSEG_VENTRICLES', 'extracted', 'Images')

    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if '.nii.gz' in f]

    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)


        label_file = file.replace('/Images/', '/Masks/')
        dest_label_path = os.path.join(converted_path, 'labels', 'ventricles', filename)
        shutil.copy(label_file, dest_label_path)

        return filename, {'image': file, 'ventricles': label_file}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing MEDSEG_VENTRICLES images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

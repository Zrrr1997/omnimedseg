from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_resect():
    converted_path = os.path.join(data_root(), 'MRI', 'RESECT', 'converted')

    # Create label mapping
    label_classes = ['brain_tumor']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'RESECT', 'extracted')

    image_paths = [os.path.join(extracted_base, el) for el in os.listdir(os.path.join(extracted_base)) if '.nii' in el]

    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        temp_dict = {'image': file}

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)

        case_id = os.path.basename(file).split('-')[0]
        label_file = os.path.join(os.path.dirname(file), 'RESECT-Segmentation', case_id, f'{case_id}-FLAIR-tumor.nii.gz') 
        shutil.copy(label_file, os.path.join(converted_path, 'labels', 'brain_tumor', filename))

        return filename, {'image': file, 'brain_tumor': label_file}


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing RESECT images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict
 

    save_mapping(converted_path, filename_map)

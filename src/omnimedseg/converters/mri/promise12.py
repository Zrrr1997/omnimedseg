from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_promise12():
    converted_path = os.path.join(data_root(), 'MRI', 'PROMISE12', 'converted')

    # Create label mapping
    label_classes = ['prostate']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'PROMISE12', 'extracted')

    image_paths = []
    image_paths += [os.path.join(extracted_base, 'livechallenge_test_data', el) for el in os.listdir(os.path.join(extracted_base, 'livechallenge_test_data')) if '.mhd' in el and 'segmentation.mhd' not in el]
    image_paths += [os.path.join(extracted_base, 'test_data', el) for el in os.listdir(os.path.join(extracted_base, 'test_data')) if '.mhd' in el and 'segmentation.mhd' not in el]
    image_paths += [os.path.join(extracted_base, 'training_data', el) for el in os.listdir(os.path.join(extracted_base, 'training_data')) if '.mhd' in el and 'segmentation.mhd' not in el]


    filename_map = {}
    split_map = {'train': [], 'test': [], 'livechallenge_test': []}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        
        convert_mhd_raw_to_nifti(file, output_path=dest_image_path)
        label_file = file.replace('.mhd', '_segmentation.mhd')
        dest_label_path = os.path.join(converted_path, 'labels', 'prostate', filename)
        convert_mhd_raw_to_nifti(label_file, output_path=dest_label_path)



        return filename, {'image': file, 'prostate': label_file}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing PICAI images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict
            if 'livechallenge' in filename_map[filename]['image']:
                split_map['livechallenge_test'].append(filename)
            elif 'test_data' in filename_map[filename]['image']:
                split_map['test'].append(filename)
            else:
                split_map['train'].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')
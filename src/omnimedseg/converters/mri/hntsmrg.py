import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import SimpleITK as sitk
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_hntsmrg():
    converted_path = os.path.join('datasets', 'MRI', 'HNTSMRG', 'converted')

    # Create label mapping
    label_classes = ['tumor', 'metastatic_lymph_node']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'HNTSMRG', 'extracted', 'HNTSMRG24_train')

    image_paths = []
    image_paths += [os.path.join(extracted_base, f, 'midRT', f'{f}_midRT_T2.nii.gz') for f in os.listdir(extracted_base)]
    image_paths += [os.path.join(extracted_base, f, 'preRT', f'{f}_preRT_T2.nii.gz') for f in os.listdir(extracted_base)]

    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)


        label_file = file.replace('T2.nii.gz', 'mask.nii.gz')
        label = sitk.ReadImage(label_file)
        tumor = sitk.Equal(label, float(1), 0, 1)
        lymph_node = sitk.Equal(label, float(2), 0, 1)
        tumor = sitk.Cast(tumor, sitk.sitkUInt8)
        lymph_node = sitk.Cast(lymph_node, sitk.sitkUInt8)

        sitk.WriteImage(tumor, os.path.join(converted_path, 'labels', 'tumor', filename))
        sitk.WriteImage(lymph_node, os.path.join(converted_path, 'labels', 'metastatic_lymph_node', filename))

        return filename, {'image': file, 'tumor': label_file, 'metastatic_lymph_node': label_file}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing HNTSMRG images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

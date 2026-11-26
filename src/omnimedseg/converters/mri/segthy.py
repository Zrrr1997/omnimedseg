from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_segthy():
    converted_path = os.path.join(data_root(), 'MRI', 'SEGTHY', 'converted')

    # Create label mapping
    label_classes = ['thyroid', 'left_carotid', 'right_carotid', 'left_jugular_vein', 'right_jugular_vein']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'SEGTHY', 'extracted', 'MRI_volunteer_dataset', 'MRI')

    image_paths = [os.path.join(extracted_base, el) for el in os.listdir(os.path.join(extracted_base)) if '.nii.gz' in el]


    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)

        label_file = file.replace('MRI_volunteer_dataset/MRI', 'MRI_volunteer_dataset/MRI_thyroid+jugular+carotid_label').replace('.nii.gz', '_thyroid+jugular+carotid_label.nii.gz')
        if not os.path.exists(label_file): # only thyroid
            label_file = label_file.replace('MRI_thyroid+jugular+carotid_label', 'MRI_thyroid_label').replace('_thyroid+jugular+carotid_label.nii.gz', '_thyroid_label.nii.gz')
            shutil.copy(label_file, os.path.join(converted_path, 'labels', 'thyroid', filename))
            return filename, {'image': file, 'thyroid': label_file}
        else: 
            label = sitk.ReadImage(label_file)
            thyroid =  sitk.Cast(sitk.Equal(label, float(1), 0, 1), sitk.sitkUInt8)
            left_carotid =  sitk.Cast(sitk.Equal(label, float(2), 0, 1), sitk.sitkUInt8)
            right_carotid =  sitk.Cast(sitk.Equal(label, float(4), 0, 1), sitk.sitkUInt8)
            left_jugular =  sitk.Cast(sitk.Equal(label, float(3), 0, 1), sitk.sitkUInt8)
            right_jugular =  sitk.Cast(sitk.Equal(label, float(5), 0, 1), sitk.sitkUInt8)
            sitk.WriteImage(thyroid, os.path.join(converted_path, 'labels', 'thyroid', filename))
            sitk.WriteImage(left_carotid, os.path.join(converted_path, 'labels', 'left_carotid', filename))
            sitk.WriteImage(right_carotid, os.path.join(converted_path, 'labels', 'right_carotid', filename))
            sitk.WriteImage(left_jugular, os.path.join(converted_path, 'labels', 'left_jugular_vein', filename))
            sitk.WriteImage(right_jugular, os.path.join(converted_path, 'labels', 'right_jugular_vein', filename))
            return filename, {'image': file, 'thyroid': label_file, 'left_carotid': label_file, 'right_carotid': label_file, 'left_jugular_vein': label_file, 'right_jugular_vein': label_file}



    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing SEGTHY images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict


    save_mapping(converted_path, filename_map)

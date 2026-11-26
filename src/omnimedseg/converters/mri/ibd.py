from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import SimpleITK as sitk
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_ibd():
    converted_path = os.path.join(data_root(), 'MRI', 'IBD', 'converted')

    # Create label mapping
    label_classes = ['stomach', 'duodenum', 'small_intestine', 'appendix', 'cecum', 'ascending_colon', 'transverse_colon', 'descending_colon', 'sigmoid_colon', 'rectum']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'IBD', 'extracted')

    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if 'data.nii.gz' in f]

    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)


        label_file = file.replace('_data.nii.gz', '_label.nii.gz')
        label = sitk.ReadImage(label_file)
        temp_dict = {'image': file}
        for i, class_label in enumerate(label_classes):
            target = sitk.Equal(label, float(i+1), 0, 1)  
            target = sitk.Cast(target, sitk.sitkUInt8)
            sitk.WriteImage(target, os.path.join(converted_path, 'labels', class_label, filename))
            temp_dict[class_label] = label_file


        return filename, temp_dict

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing IBD images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

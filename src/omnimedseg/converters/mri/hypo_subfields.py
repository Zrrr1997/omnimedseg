import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_hypo_subfields():
    converted_path = os.path.join('datasets', 'MRI', 'HYPO_SUBFIELDS', 'converted')

    # Create label mapping
    label_classes = ['left_CA2', 'left_CA1', 'left_CA4&DG', 'left_ERC', 'left_CA3', 'left_SUB', 'left_hippocampus_tail']
    right_classes = [el.replace('left', 'right') for el in label_classes]
    label_classes += right_classes
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'HYPO_SUBFIELDS', 'extracted', 'hippo_subfield', '7T_T2w_0.7_for_subfield_delineation')

    image_paths = [os.path.join(extracted_base, el) for el in os.listdir(os.path.join(extracted_base)) if '.nii' in el]


    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        image = sitk.ReadImage(file)
        sitk.WriteImage(image, dest_image_path)

        label_file_left = file.replace('7T_T2w_0.7_for_subfield_delineation', 'hippo_label').replace('t2_s', 'sub-').replace('_0.7.nii', '-L.nii.gz') 
        label_file_right = file.replace('7T_T2w_0.7_for_subfield_delineation', 'hippo_label').replace('t2_s', 'sub-').replace('_0.7.nii', '-R.nii.gz') 

        temp_dict = {'image': file}
        for i, label_file in enumerate([label_file_left, label_file_right]):
            label_img = sitk.ReadImage(label_file)
            sub = sitk.BinaryThreshold(label_img, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
            ca2 = sitk.BinaryThreshold(label_img, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)
            ca1 = sitk.BinaryThreshold(label_img, lowerThreshold=3, upperThreshold=3, insideValue=1, outsideValue=0)
            ca4dg = sitk.BinaryThreshold(label_img, lowerThreshold=4, upperThreshold=4, insideValue=1, outsideValue=0)
            erc = sitk.BinaryThreshold(label_img, lowerThreshold=5, upperThreshold=5, insideValue=1, outsideValue=0)
            ca3 = sitk.BinaryThreshold(label_img, lowerThreshold=6, upperThreshold=6, insideValue=1, outsideValue=0)
            tail = sitk.BinaryThreshold(label_img, lowerThreshold=7, upperThreshold=7, insideValue=1, outsideValue=0)

            loc = 'left' if i == 0 else 'right'

            sitk.WriteImage(sub, os.path.join(converted_path, 'labels', f'{loc}_SUB', filename))
            sitk.WriteImage(ca2, os.path.join(converted_path, 'labels', f'{loc}_CA2', filename))
            sitk.WriteImage(ca1, os.path.join(converted_path, 'labels', f'{loc}_CA1', filename))
            sitk.WriteImage(ca4dg, os.path.join(converted_path, 'labels', f'{loc}_CA4&DG', filename))
            sitk.WriteImage(erc, os.path.join(converted_path, 'labels', f'{loc}_ERC', filename))
            sitk.WriteImage(ca3, os.path.join(converted_path, 'labels', f'{loc}_CA3', filename))
            sitk.WriteImage(tail, os.path.join(converted_path, 'labels', f'{loc}_hippocampus_tail', filename))
            for el in label_classes:
                if loc in el:
                    temp_dict[el] = label_file


        return filename, temp_dict

    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing HYPO_SUBFIELDS images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict


    save_mapping(converted_path, filename_map)

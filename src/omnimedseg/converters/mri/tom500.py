from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_tom500():
    converted_path = os.path.join(data_root(), 'MRI', 'TOM500', 'converted')

    # Create label mapping
    label_classes = ['superior_rectus', 'superior_oblique', 'lacrimal_gland', 'optic_nerve', 'lateral_rectus', 'medial_rectus', 'eyeball', 'orbital_fat', 'inferior_rectus']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'TOM500', 'extracted', 'TOM500')

    image_paths = []
    image_paths += [os.path.join(extracted_base, 'train', 'image', el) for el in os.listdir(os.path.join(extracted_base, 'train', 'image')) if '.nii' in el]
    image_paths += [os.path.join(extracted_base, 'val', 'image', el) for el in os.listdir(os.path.join(extracted_base, 'val', 'image')) if '.nii' in el]

    filename_map = {}
    split_map = {'train': [], 'val': []}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        temp_dict = {'image': file}

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)

        label_file = file.replace('/image/', '/label/') 

        label_img = sitk.ReadImage(label_file)
        eb = sitk.BinaryThreshold(label_img, lowerThreshold=9, upperThreshold=9, insideValue=1, outsideValue=0)
        of = sitk.BinaryThreshold(label_img, lowerThreshold=6, upperThreshold=6, insideValue=1, outsideValue=0)
        on = sitk.BinaryThreshold(label_img, lowerThreshold=5, upperThreshold=5, insideValue=1, outsideValue=0)
        lg = sitk.BinaryThreshold(label_img, lowerThreshold=7, upperThreshold=7, insideValue=1, outsideValue=0)
        lr = sitk.BinaryThreshold(label_img, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)
        sr = sitk.BinaryThreshold(label_img, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
        so = sitk.BinaryThreshold(label_img, lowerThreshold=8, upperThreshold=8, insideValue=1, outsideValue=0)
        ir = sitk.BinaryThreshold(label_img, lowerThreshold=4, upperThreshold=4, insideValue=1, outsideValue=0)
        mr = sitk.BinaryThreshold(label_img, lowerThreshold=3, upperThreshold=3, insideValue=1, outsideValue=0)

        sitk.WriteImage(eb, os.path.join(converted_path, 'labels', f'eyeball', filename))
        sitk.WriteImage(of, os.path.join(converted_path, 'labels', f'orbital_fat', filename))
        sitk.WriteImage(on, os.path.join(converted_path, 'labels', f'optic_nerve', filename))
        sitk.WriteImage(lg, os.path.join(converted_path, 'labels', f'lacrimal_gland', filename))
        sitk.WriteImage(lr, os.path.join(converted_path, 'labels', f'lateral_rectus', filename))
        sitk.WriteImage(sr, os.path.join(converted_path, 'labels', f'superior_rectus', filename))
        sitk.WriteImage(so, os.path.join(converted_path, 'labels', f'superior_oblique', filename))
        sitk.WriteImage(ir, os.path.join(converted_path, 'labels', f'inferior_rectus', filename))
        sitk.WriteImage(mr, os.path.join(converted_path, 'labels', f'medial_rectus', filename))

        for el in label_classes:
            temp_dict[el] = label_file
        return filename, temp_dict


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing TOM500 images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict
            if 'train' in filename_map[filename]['image']:
                split_map['train'].append(filename)
            else:
                split_map['val'].append(filename)


    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')

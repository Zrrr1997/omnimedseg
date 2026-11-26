from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
from collections import defaultdict

def convert_acdc():
    converted_path = os.path.join(data_root(), 'MRI', 'ACDC', 'converted')

    # Create label mapping
    label_classes = ['right_ventricle', 'myocardium', 'left_ventricle']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'ACDC', 'extracted', 'ACDC', 'database')

    image_paths = []
    for split in ['training', 'testing']:
        patients = [os.path.join(extracted_base, split, f) for f in os.listdir(os.path.join(extracted_base, split)) if '.md' not in f]
        for patient in patients:
            image_paths += [os.path.join(patient, el) for el in os.listdir(patient) if 'frame' in el and '_gt' not in el]

    filename_map = {}
    split_map = defaultdict(list)


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"
        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)
        label_classes = ['right_ventricle', 'myocardium', 'left_ventricle']

        label_file = file.replace('.nii.gz', '_gt.nii.gz')
        label = sitk.ReadImage(label_file)
        temp_dict = {'image': file}

        for label_id in np.unique(sitk.GetArrayFromImage(label)):
            if label_id == 0:
                continue
            label_image = sitk.Equal(label, float(label_id), 0, 1)
            label_image = sitk.Cast(label_image, sitk.sitkUInt8)

            out_file = os.path.join(converted_path, 'labels', label_classes[label_id-1], filename)
            sitk.WriteImage(label_image, out_file)
            temp_dict[label_classes[label_id-1]] = label_file

        if 'training' in file:
            split = 'train'
        elif 'testing' in file:
            split = 'test'


        return filename, temp_dict, split

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing ACDC images"):
            filename, temp_dict, split = f.result()
            filename_map[filename] = temp_dict
            split_map[split].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, dict(split_map), json_fn='split_map')
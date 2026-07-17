import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils.dataset_utils import *
import SimpleITK as sitk
import numpy as np

def _process_ribseg_v2(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    data = {'image': file}
    split = 'test'
    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)
    fracture_label_classes = ['displaced_rib_fracture', 'non_displaced_rib_fracture', 'buckle_rib_fracture', 'segmental_rib_fracture']
    if 'Part1' in file or 'Part2' in file or 'ribfrac-val-images' in file:
        fractures = file.replace('image.nii.gz', 'label.nii.gz').replace('ribfrac-val-images', 'ribfrac-val-labels')
        label_img = sitk.ReadImage(fractures)
        for i, label in enumerate(fracture_label_classes):
            dst_label = os.path.join(converted_path, 'labels', label, filename)
            label_i = sitk.BinaryThreshold(label_img, lowerThreshold=i+1, upperThreshold=i+1, insideValue=1, outsideValue=0)
            if np.any(sitk.GetArrayViewFromImage(label_i)):
                sitk.WriteImage(label_i, dst_label)
                data[label] = fractures
        dst_label = os.path.join(converted_path, 'labels', 'rib_fracture', filename)
        label_i = sitk.BinaryThreshold(label_img, lowerThreshold=1, upperThreshold=5, insideValue=1, outsideValue=0)
        sitk.WriteImage(label_i, dst_label)
        data['rib_fracture'] = fractures
        if '-val-' in file:
            split = 'val'
        else:
            split = 'train'
    
    

    rib_label_file = os.path.join('datasets/CT/RIBSEG_V2/extracted/ribseg_v2/seg', os.path.basename(file.replace('-image.nii.gz', '-rib-seg.nii.gz')))
    dst_label = os.path.join(converted_path, 'labels', 'ribs', filename)
    label_img = sitk.ReadImage(rib_label_file)
    label_i = sitk.BinaryThreshold(label_img, lowerThreshold=1, upperThreshold=35, insideValue=1, outsideValue=0)
    sitk.WriteImage(label_i, dst_label)
    data['ribs'] = rib_label_file

    return filename, data, split


def convert_ribseg_v2(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'RIBSEG_V2', 'converted')
    label_classes = ['ribs', 'rib_fracture', 'displaced_rib_fracture', 'non_displaced_rib_fracture', 'buckle_rib_fracture', 'segmental_rib_fracture']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'RIBSEG_V2', 'extracted')

    image_paths = []
    for base in ['Part1', 'Part2', 'ribfrac-test-images', 'ribfrac-val-images']:
        image_paths += [os.path.join(extracted_base, base, f) for f in os.listdir(os.path.join(extracted_base, base)) if 'image.nii.gz' in f]

    filename_map = {}
    split_map = {'train': [], 'val': [], 'test':[]}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_ribseg_v2, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing RIBSEG_V2 images"):
            filename, mapping, split = future.result()
            filename_map[filename] = mapping
            split_map[split].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')

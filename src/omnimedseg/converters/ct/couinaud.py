import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
import SimpleITK as sitk
import numpy as np


def _process_couinaud(args):
    idx, file, converted_path, label_classes = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)
    data = {'image': file}
    # Copy label
    for i, dest in enumerate(['couinaud_annotation', 'liver_annotation']):
        label_file = os.path.join('datasets', 'CT', 'COUINAUD', 'extracted', 'dataset-master', dest, os.path.basename(file))
        if os.path.exists(label_file):
            if dest == 'liver_annotation':
                dst_label = os.path.join(converted_path, 'labels', 'liver', filename)
                shutil.copy(label_file, dst_label)
                data['liver'] = label_file
            else:
                for i in range(1, 9):
                    dst_label = os.path.join(converted_path, 'labels', f'couinaud_segment_{i}', filename)
                    seg_file = sitk.ReadImage(label_file)
                    seg_array = sitk.GetArrayFromImage(seg_file)
                    seg_array_bin = (seg_array == i).astype(seg_array.dtype)
                    seg_img = sitk.GetImageFromArray(seg_array_bin)
                    seg_img.CopyInformation(seg_file)
                    sitk.WriteImage(seg_img, dst_label)
                    data[f'couinaud_segment_{i}'] = label_file
    return filename, data


def convert_couinaud(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'COUINAUD', 'converted')
    label_classes = ['liver']
    for i in range(1, 9):
        label_classes.append(f'couinaud_segment_{i}')
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'COUINAUD', 'extracted')
    if not os.path.exists('datasets/CT/MSD_HEPATIC_VESSEL/extracted'):
        raise FileNotFoundError("Please download and extract the MSD_HEPATIC_VESSEL dataset first, as COUINAUD depends on it. \n You can do this by running: \n omnimedseg --dataset MSD_HEPATIC_VESSEL --download --extract")
    

    extract_archives(os.path.join(extracted_base, 'dataset-master'))

    image_paths = []
    image_paths += [os.path.join(extracted_base.replace('COUINAUD', 'MSD_HEPATIC_VESSEL'), 'Task08_HepaticVessel', 'imagesTr', f)
                    for f in os.listdir(os.path.join(extracted_base.replace('COUINAUD', 'MSD_HEPATIC_VESSEL'), 'Task08_HepaticVessel', 'imagesTr')) if f in os.listdir(os.path.join(extracted_base, 'dataset-master', 'couinaud_annotation')) or f in os.listdir(os.path.join(extracted_base, 'dataset-master', 'liver_annotation'))]
    image_paths += [os.path.join(extracted_base.replace('COUINAUD', 'MSD_HEPATIC_VESSEL'), 'Task08_HepaticVessel', 'imagesTs', f)
                    for f in os.listdir(os.path.join(extracted_base.replace('COUINAUD', 'MSD_HEPATIC_VESSEL'), 'Task08_HepaticVessel', 'imagesTs')) if f in os.listdir(os.path.join(extracted_base, 'dataset-master', 'couinaud_annotation')) or f in os.listdir(os.path.join(extracted_base, 'dataset-master', 'liver_annotation'))]    

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_couinaud, (i, file, converted_path, label_classes)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing COUINAUD images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

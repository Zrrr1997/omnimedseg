import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import SimpleITK as sitk
from src.utils.dataset_utils import *


def _process_kits23(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)

    # Copy label
    label_file = file.replace('imaging.nii.gz', 'segmentation.nii.gz')
    label_img = sitk.ReadImage(label_file)
    label_kidneys = sitk.Equal(label_img, float(1), 0, 1)
    label_kidney_tumor = sitk.BinaryThreshold(label_img, lowerThreshold=2, upperThreshold=10, insideValue=1, outsideValue=0)

    label_kidneys = sitk.Cast(label_kidneys, sitk.sitkUInt8)
    label_kidney_tumor = sitk.Cast(label_kidney_tumor, sitk.sitkUInt8)

    dst_label = os.path.join(converted_path, 'labels', 'kidneys', filename)
    sitk.WriteImage(label_kidneys, dst_label)
    dst_label = os.path.join(converted_path, 'labels', 'kidney_tumor', filename)
    sitk.WriteImage(label_kidney_tumor, dst_label)
    return filename, {
        'image': file,
        'kidney': label_file,
        'kidney_tumor': label_file
    }


def convert_kits23(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'KITS23', 'converted')
    label_classes = ['kidneys', 'kidney_tumor']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'KITS23', 'extracted', 'kits23-main', 'dataset')
    image_paths = [os.path.join(extracted_base, f, 'imaging.nii.gz') for f in os.listdir(os.path.join(extracted_base)) if 'case' in f]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_kits23, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing KITS23 images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

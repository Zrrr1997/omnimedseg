from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, converted_path, idx):
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label = cv2.imread(file.replace('.png', '_mask.png'))

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'breast_tumor', filename), label)

    return filename, {
        'image': file,
        'breast_tumor': file.replace('.png', '_mask.png'),
    }

def convert_busi():
    converted_path = os.path.join(data_root(), 'Ultrasound', 'BUSI', 'converted')
    ensure_directories(converted_path, ['breast_tumor'])
    filename_map = {}

    extracted_base = os.path.join(data_root(), 'Ultrasound', 'BUSI', 'extracted', 'Dataset_BUSI_with_GT')

    image_files_benign = [os.path.join(extracted_base, 'benign', f) for f in os.listdir(os.path.join(extracted_base, 'benign')) if 'mask' not in f]
    image_files_malignant = [os.path.join(extracted_base, 'malignant', f) for f in os.listdir(os.path.join(extracted_base, 'malignant')) if 'mask' not in f]
    image_files_normal = [os.path.join(extracted_base, 'normal', f) for f in os.listdir(os.path.join(extracted_base, 'normal')) if 'mask' not in f]
    image_files = image_files_benign + image_files_malignant + image_files_normal

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, file, converted_path, idx): idx for idx, file in enumerate(image_files)}
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Processing BUSI images"):
            filename, file_map = future.result()
            filename_map[filename] = file_map

    save_mapping(converted_path, filename_map)

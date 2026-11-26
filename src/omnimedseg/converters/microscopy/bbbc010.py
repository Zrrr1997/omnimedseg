from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(idx, file, image_files_2, extracted_base, converted_path):
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    label_index = os.path.basename(file).split('_')[6]

    file_2 = [f for f in image_files_2 if label_index in f][0]
    image_2 = cv2.imread(file_2)
    image_2 = cv2.normalize(image_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    label_file = os.path.join(extracted_base, 'BBBC010_v1_foreground', f'{label_index}_binary.png')
    label = cv2.imread(label_file)

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'images_2', filename), image_2)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'c_elegans', filename), label)

    return filename, {
        'image': file,
        'image_2': file_2,
        'c_elegans': label_file,
    }

def convert_bbbc010():
    converted_path = os.path.join(data_root(), 'Microscopy', 'BBBC010', 'converted')
    ensure_directories(converted_path, ['c_elegans'])
    os.makedirs(os.path.join(converted_path, 'images_2'), exist_ok=True)

    extracted_base = os.path.join(data_root(), 'Microscopy', 'BBBC010', 'extracted')
    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if '_w1_' in f]
    image_files_2 = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if '_w2_' in f]

    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, idx, file, image_files_2, extracted_base, converted_path)
            for idx, file in enumerate(image_files)
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing BBBC010 images"):
            filename, mapping = f.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
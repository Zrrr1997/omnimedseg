from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(idx_image_path):
    idx, image_path, converted_path = idx_image_path
    if not os.path.exists(os.path.join(image_path, 'masks')):
        return None

    image_file = [os.path.join(image_path, 'images', f) for f in os.listdir(os.path.join(image_path, 'images'))][0]
    label_files = [os.path.join(image_path, 'masks', f) for f in os.listdir(os.path.join(image_path, 'masks'))]
    image = cv2.imread(image_file)
    label = np.zeros(image.shape).astype(np.uint8)
    for label_file in label_files:
        label += cv2.imread(label_file).astype(np.uint8)

    filename = f"{idx:06d}.png"
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus', filename), label)

    return filename, {
        'image': image_file,
        'nucleus': label_files
    }

def convert_bbbc038():
    converted_path = os.path.join(data_root(), 'Microscopy', 'BBBC038', 'converted')
    ensure_directories(converted_path, ['nucleus'])

    extracted_base = os.path.join(data_root(), 'Microscopy', 'BBBC038', 'extracted')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]
    filename_map = {}

    args = [(idx, image_path, converted_path) for idx, image_path in enumerate(image_paths)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, arg) for arg in args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing BBBC038 images"):
            result = f.result()
            if result is not None:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
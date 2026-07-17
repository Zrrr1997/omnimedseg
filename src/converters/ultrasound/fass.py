import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def process_file(file, converted_path, idx):
    filename = f"{idx:06d}.png"
    np_data = np.load(file, allow_pickle=True).item()

    image = np_data['image']
    labels = np_data['structures']

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'aorta', filename), labels['artery'] * 255)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'umbilical_vein', filename), labels['vein'] * 255)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'liver', filename), labels['liver'] * 255)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'stomach', filename), labels['stomach'] * 255)

    return filename, {k: file for k in ['image', 'aorta', 'umbilical_vein', 'liver', 'stomach']}

def convert_fass():
    converted_path = os.path.join('datasets', 'Ultrasound', 'FASS', 'converted')
    ensure_directories(converted_path, ['aorta', 'liver', 'stomach', 'umbilical_vein'])
    extracted_base = os.path.join('datasets', 'Ultrasound', 'FASS', 'extracted', 'ARRAY_FORMAT')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]
    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, converted_path, idx): idx for idx, file in enumerate(image_files)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing FASS images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

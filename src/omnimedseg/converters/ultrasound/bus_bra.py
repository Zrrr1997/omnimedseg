from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_file(file, extracted_base, converted_path):
    try:
        filename = f"{os.path.basename(file).split('.')[0]}.png"

        image = cv2.imread(file)
        label = cv2.imread(file.replace('Images', 'Masks').replace('bus_', 'mask_'))

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'breast_tumor', filename), label)
        return filename, {
            'image': file,
            'breast_tumor': file.replace('Images', 'Masks')
        }
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def convert_bus_bra():
    converted_path = os.path.join(data_root(), 'Ultrasound', 'BUS_BRA', 'converted')
    ensure_directories(converted_path, ['breast_tumor'])
    filename_map = {}
    extracted_base = os.path.join(data_root(), 'Ultrasound', 'BUS_BRA', 'extracted', 'BUSBRA', 'Images')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, extracted_base, converted_path) for file in image_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing BUS_BRA images"):
            result = future.result()
            if result:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
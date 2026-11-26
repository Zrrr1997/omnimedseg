from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, converted_path):
    try:
        filename = f"{os.path.splitext(os.path.basename(file))[0]}.png"

        image = cv2.imread(file)
        label = cv2.imread(file.replace('images', 'masks'))
        label = np.where(np.any(label > 0, axis=-1), 255, 0).astype(np.uint8)

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'breast_tumor', filename), label)
        return filename, {
            'image': file,
            'breast_tumor': file.replace('images', 'masks')
        }
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def convert_bus_uclm():
    converted_path = os.path.join(data_root(), 'Ultrasound', 'BUS_UCLM', 'converted')
    ensure_directories(converted_path, ['breast_tumor'])
    filename_map = {}
    extracted_base = os.path.join(data_root(), 'Ultrasound', 'BUS_UCLM', 'extracted', 'BUS-UCLM Breast ultrasound lesion segmentation dataset', 'BUS-UCLM', 'images')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file, converted_path) for file in image_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing BUS_UCLM images"):
            result = future.result()
            if result:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
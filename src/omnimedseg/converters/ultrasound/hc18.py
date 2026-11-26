from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, converted_path):
    try:
        filename = os.path.basename(file)
        filename = f"{os.path.splitext(filename)[0]}.png"

        image = cv2.imread(file)
        label = cv2.imread(file.replace('.png', '_Annotation.png'))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill the contours
        cv2.drawContours(label, contours, -1, color=255, thickness=cv2.FILLED)
        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'fetal_head', filename), label)

        return filename, {
            'image': file,
            'fetal_head': file.replace('.png', '_Annotation.png')
        }
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def convert_hc18():
    converted_path = os.path.join(data_root(), 'Ultrasound', 'HC18', 'converted')
    ensure_directories(converted_path, ['fetal_head'])
    filename_map = {}
    extracted_base = os.path.join(data_root(), 'Ultrasound', 'HC18', 'extracted', 'training_set')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if 'Annotation' not in f]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file, converted_path) for file in image_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing HC18 images"):
            result = future.result()
            if result:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
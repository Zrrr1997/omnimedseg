import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def load_class_mapping(classes_json_path):
    with open(classes_json_path, "r") as f:
        config = json.load(f)
    return {
        class_def["title"]: idx + 1  # start at 1 to avoid background=0
        for idx, class_def in enumerate(config.get("classes", []))
        if class_def.get("shape") == "bitmap"
    }

def process_image(file, idx, converted_path, class_mapping):
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('/img/', '/ann/').replace('.jpeg', '.jpeg.json')
    label = np.array(json_mask_to_png(label_file, class_mapping))
    label = (label > 0).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'intraretinal_fluid', filename), label)
    return filename, file, label_file

def convert_intraretinal_cystoid_fluid():
    converted_path = os.path.join('datasets', 'OCT', 'INTRARETINAL_CYSTOID_FLUID', 'converted')
    ensure_directories(converted_path, ['intraretinal_fluid'])
    filename_map = {}

    extracted_base = os.path.join('datasets', 'OCT', 'INTRARETINAL_CYSTOID_FLUID', 'extracted', 'ds', 'img')
    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]
    class_mapping = load_class_mapping(os.path.join('datasets', 'OCT', 'INTRARETINAL_CYSTOID_FLUID', 'extracted', 'meta.json'))

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, file, idx, converted_path, class_mapping): idx
            for idx, file in enumerate(image_files)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing INTRARETINAL_CYSTOID_FLUID images"):
            idx = futures[future]
            try:
                filename, file, label_file = future.result()
                filename_map[filename] = {
                    'image': file,
                    'intraretinal_fluid': label_file,
                }
            except Exception as e:
                print(f"Error processing file {image_files[idx]}: {e}")

    save_mapping(converted_path, filename_map)
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, idx, converted_path, class_mapping):
    filename = f"{idx:06d}.png"

    image = cv2.imread(file)
    label_file = file.replace('img', 'ann').replace('.jpg', '.jpg.json')
    label = np.array(json_mask_to_png(label_file, class_mapping))

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'tooth', filename), (label > 0).astype(np.uint8) * 255)

    return filename, {'image': file, 'tooth': label_file}

def convert_teeth_seg():
    converted_path = os.path.join('datasets', 'XRay', 'TEETH_SEG', 'converted')
    ensure_directories(converted_path, ['tooth'])

    extracted_base = os.path.join('datasets', 'XRay', 'TEETH_SEG', 'extracted', 'ds', 'img')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    class_mapping = load_class_mapping(os.path.join('datasets', 'XRay', 'TEETH_SEG', 'extracted', 'meta.json'))

    filename_map = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, file, idx, converted_path, class_mapping): idx
            for idx, file in enumerate(image_paths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing TEETH_SEG images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
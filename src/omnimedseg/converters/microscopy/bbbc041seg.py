
from omnimedseg.core.paths import data_root
import os
import cv2
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def convert_bbbc041seg():


    def process_image(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.png"
        with suppress_stderr():
            image = cv2.imread(file)
            label = cv2.imread(file.replace('Original', 'Mask').replace('.jpg', '.png'))

        if image is None or label is None:
            return None  # Skip if image or label failed to load

        image_path = os.path.join(converted_path, 'images', filename)
        label_path = os.path.join(converted_path, 'labels', 'blood_cell', filename)

        cv2.imwrite(image_path, image)
        cv2.imwrite(label_path, label)

        return filename, {
            'image': file,
            'blood_cell': file.replace('Original', 'Mask').replace('.jpg', '.png')
        }

    converted_path = os.path.join(data_root(), 'Microscopy', 'BBBC041SEG', 'converted')
    ensure_directories(converted_path, ['blood_cell'])

    extracted_base = os.path.join(
        data_root(), 'Microscopy', 'BBBC041SEG', 'extracted',
        'Final data', 'Final data', 'New folder', 'Original'
    )

    image_paths = [
        os.path.join(extracted_base, f) for f in os.listdir(extracted_base)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(process_image, (idx, path)) for idx, path in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing BBBC041SEG images"):
            result = future.result()
            if result:
                filename, entry = result
                filename_map[filename] = entry

    save_mapping(converted_path, filename_map)
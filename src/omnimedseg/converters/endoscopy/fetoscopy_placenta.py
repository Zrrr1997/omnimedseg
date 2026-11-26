import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, idx, converted_path):
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('/images/', '/masks_gt/').replace('.png', '_mask.png')
    label = (cv2.imread(label_file) > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'placental_vessels', filename), label)

    return filename, {
        'image': file,
        'placental_vessels': label_file,
    }

def convert_fetoscopy_placenta():
    converted_path = os.path.join('datasets', 'Endoscopy', 'FETOSCOPY_PLACENTA', 'converted')
    ensure_directories(converted_path, ['placental_vessels'])
    filename_map = {}

    extracted_base = os.path.join('datasets', 'Endoscopy', 'FETOSCOPY_PLACENTA', 'extracted', 'Vessel_segmentation_annotations')

    image_files = []
    for image_path in os.listdir(extracted_base):
        if image_path.startswith('.'):
            continue
        image_dir = os.path.join(extracted_base, image_path, 'images')
        image_files.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir)])

    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, file in enumerate(image_files):
            futures.append(executor.submit(process_image, file, idx, converted_path))

        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing FETOSCOPY_PLACENTA images"):
            filename, mapping = f.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
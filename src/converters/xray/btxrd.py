import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.dataset_utils import *
import json

def process_file(idx_file, converted_path):
    idx, file = idx_file
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)

    label_path = file.replace('images', 'Annotations').replace('jpeg', 'json').replace('jpg', 'json')
    if not os.path.exists(label_path):
        return None

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)

    with open(label_path, 'r') as f:
        label_json = json.load(f)
    label = np.zeros(image.shape[:2], dtype=np.uint8)
    for shape in label_json['shapes']:
        contour = np.array(shape['points'])
        shape_dims = (label_json['imageHeight'], label_json['imageWidth'], 3)
        label += (contour_to_mask(contour, shape_dims) > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'labels', 'bone_tumor', filename), label)

    return filename, {
        'image': file,
        'bone_tumor': label_path
    }

def convert_btxrd():
    converted_path = os.path.join('datasets', 'XRay', 'BTXRD', 'converted')
    ensure_directories(converted_path, ['bone_tumor'])

    extracted_base = os.path.join('datasets', 'XRay', 'BTXRD', 'extracted', 'BTXRD', 'images')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    filename_map = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(process_file, (idx, file), converted_path) for idx, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing BTXRD images"):
            result = future.result()
            if result:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

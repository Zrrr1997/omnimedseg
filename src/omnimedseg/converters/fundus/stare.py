from omnimedseg.core.paths import data_root
import os 
import cv2
import gzip
import numpy as np
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def load_ppm_gz(filepath):
    with gzip.open(filepath, 'rb') as f:
        buffer = f.read()
    # Convert binary buffer to numpy array
    img_array = np.frombuffer(buffer, dtype=np.uint8)
    # Decode image using OpenCV
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    return img

def convert_stare():
    converted_path = os.path.join(data_root(), 'Fundus', 'STARE', 'converted')
    ensure_directories(converted_path, ['vessels'])
    idx = 0
    filename_map = {}

    extracted_base = os.path.join(data_root(), 'Fundus', 'STARE', 'extracted')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if 'ah' not in f and 'vk' not in f]
    os.makedirs(os.path.join(converted_path, 'labels', 'vessels_annotator_1'), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'labels', 'vessels_annotator_2'), exist_ok=True)

    for file in tqdm(image_files, desc="Processing STARE images"):
        filename = f"{idx:06d}.png"

        image = load_ppm_gz(file)
        label_ah = load_ppm_gz(file.replace('.ppm', '.ah.ppm'))
        label_vk = load_ppm_gz(file.replace('.ppm', '.vk.ppm'))
        label = (((label_ah > 0) + (label_vk > 0)) > 0).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels_annotator_1', filename), label_ah)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels_annotator_2', filename), label_vk)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels', filename), label)
        filename_map[filename] = {
            'image': file,
            'vessels_annotator_1': file.replace('.ppm', '.ah.ppm'),
            'vessels_annotator_2': file.replace('.ppm', '.vk.ppm'),
            'vessels': [file.replace('.ppm', '.ah.ppm'), file.replace('.ppm', '.vk.ppm')]
        }
        idx += 1
    save_mapping(converted_path, filename_map)

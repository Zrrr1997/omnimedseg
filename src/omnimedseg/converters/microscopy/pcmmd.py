from omnimedseg.core.paths import data_root
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('images', 'masks').replace('.jpg', '.json')
    if not os.path.exists(label_file):
        return None
    with open(label_file, 'r') as f:
        label_json = json.load(f)
    if len(label_json['shapes']) == 0:
        return None

    contour = np.array(label_json['shapes'][0]['points'])
    shape = (label_json['imageHeight'], label_json['imageWidth'], 3)
    label = (contour_to_mask(contour, shape) > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'plasma_cell', filename), label)
    return (filename, {'image': file, 'plasma_cell': label_file})

def convert_pcmmd():
    converted_path = os.path.join(data_root(), 'Microscopy', 'PCMMD', 'converted')
    ensure_directories(converted_path, ['plasma_cell'])

    extracted_base = os.path.join(data_root(), 'Microscopy', 'PCMMD', 'extracted', 'PCMMD Plasma Cells for Multiple Myeloma Diagnosis', 'data', 'segmentation')
    image_files = []
    image_files += [os.path.join(extracted_base, 'non-plasma cells', 'images', f) for f in os.listdir(os.path.join(extracted_base, 'non-plasma cells', 'images'))]
    image_files += [os.path.join(extracted_base, 'plasma cells', 'images', f) for f in os.listdir(os.path.join(extracted_base, 'plasma cells', 'images'))]

    filename_map = {}

    args_list = [(idx, file, converted_path) for idx, file in enumerate(image_files)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing PCMMD images"):
            result = f.result()
            if result is not None:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
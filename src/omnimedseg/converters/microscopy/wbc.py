from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label = cv2.imread(file.replace('.bmp', '.png'))

    cytoplasm = (label == 128).astype(np.uint8) * 255
    nucleus = (label == 255).astype(np.uint8) * 255
    cell = (label > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus', filename), nucleus)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'cytoplasm', filename), cytoplasm)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'white_blood_cell', filename), cell)

    return filename, {
        'image': file,
        **{k: file.replace('.bmp', '.png') for k in ['nucleus', 'cytoplasm', 'white_blood_cell']}
    }

def convert_wbc():
    converted_path = os.path.join(data_root(), 'Microscopy', 'WBC', 'converted')
    ensure_directories(converted_path, ['nucleus', 'cytoplasm', 'white_blood_cell'])

    extracted_base = os.path.join(data_root(), 'Microscopy', 'WBC', 'extracted', 'segmentation_WBC-master')
    image_files = []
    image_files += [os.path.join(extracted_base, 'Dataset 1', f) for f in os.listdir(os.path.join(extracted_base, 'Dataset 1')) if '.bmp' in f]
    image_files += [os.path.join(extracted_base, 'Dataset 2', f) for f in os.listdir(os.path.join(extracted_base, 'Dataset 2')) if '.bmp' in f]

    filename_map = {}

    args_list = [(idx, file, converted_path) for idx, file in enumerate(image_files)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing WBC images"):
            filename, mapping = f.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
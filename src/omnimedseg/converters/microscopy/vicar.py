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
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    label_path = file.replace('img.tif', 'mask.png')
    label = (cv2.imread(label_path) > 0).astype(np.uint8) * 255

    image_out = os.path.join(converted_path, 'images', filename)
    label_out = os.path.join(converted_path, 'labels', 'cell', filename)
    cv2.imwrite(image_out, image)
    cv2.imwrite(label_out, label)

    return filename, file, label_path

def convert_vicar():
    converted_path = os.path.join(data_root(), 'Microscopy', 'VICAR', 'converted')
    ensure_directories(converted_path, ['cell'])

    extracted_base = os.path.join(data_root(), 'Microscopy', 'VICAR', 'extracted')
    image_paths = []
    for subpath in ['A2058', 'G361', 'HOB', 'PC3', 'PNT1A']:
        image_paths += [os.path.join(extracted_base, subpath, f) for f in os.listdir(os.path.join(extracted_base, subpath)) if '.tif' in f]

    filename_map = {}

    args_list = [(idx, file, converted_path) for idx, file in enumerate(image_paths)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing VICAR images"):
            filename, file, label_path = future.result()
            filename_map[filename] = {
                'image': file,
                'cell': label_path
            }

    save_mapping(converted_path, filename_map)

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
    label = cv2.imread(file.replace('im.tif', 'mask.tif'), cv2.IMREAD_UNCHANGED)
    cell = (label > 0).astype(np.uint8) * 255

    image_path = os.path.join(converted_path, 'images', filename)
    label_path = os.path.join(converted_path, 'labels', 'yeast_cell', filename)
    cv2.imwrite(image_path, image)
    cv2.imwrite(label_path, cell)

    return filename, {
        'image': file,
        'yeast_cell': file.replace('im.tif', 'mask.tif')
    }

def convert_yeaz():
    converted_path = os.path.join(data_root(), 'Microscopy', 'YEAZ', 'converted')
    ensure_directories(converted_path, ['yeast_cell'])

    extracted_base = os.path.join(data_root(), 'Microscopy', 'YEAZ', 'extracted')
    image_files = []
    image_files += [os.path.join(extracted_base, 'gold-standard-BF-V-1', f) for f in os.listdir(os.path.join(extracted_base, 'gold-standard-BF-V-1')) if 'im.tif' in f]
    image_files += [os.path.join(extracted_base, 'gold-standard-PhC-plus-2', f) for f in os.listdir(os.path.join(extracted_base, 'gold-standard-PhC-plus-2')) if 'im.tif' in f]

    filename_map = {}

    args_list = [(idx, file, converted_path) for idx, file in enumerate(image_files)]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, args_list), total=len(args_list), desc="🖼️  Processing YEAZ images"))

    for filename, mapping in results:
        filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
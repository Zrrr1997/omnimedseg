from omnimedseg.core.paths import data_root
import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, converted_path):
    filename = f"{process_image.idx:06d}.png"
    process_image.idx += 1

    image = cv2.imread(file)
    label_file = file.replace('1.img.png', '2.mask.png')
    label = cv2.imread(label_file)

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'pneumothorax', filename), label)

    return filename, file, label_file

process_image.idx = 0

def convert_ptx_498():
    converted_path = os.path.join(data_root(), 'XRay', 'PTX_498', 'converted')
    ensure_directories(converted_path, ['pneumothorax'])

    extracted_base = os.path.join(data_root(), 'XRay', 'PTX_498', 'extracted')
    image_paths = []
    for site in ['SiteA', 'SiteB', 'SiteC']:
        image_paths += [os.path.join(extracted_base, site, f) for f in os.listdir(os.path.join(extracted_base, site)) if 'img.png' in f]

    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file, converted_path) for file in image_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing PTX_498 images"):
            filename, file, label_file = future.result()
            filename_map[filename] = {
                'image': file,
                'pneumothorax': label_file
            }

    save_mapping(converted_path, filename_map)
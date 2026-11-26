from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *



def process_image(file, converted_path, target, idx):
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('image', 'mask')
    label = cv2.imread(label_file)
    label = (label > 240).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', target, filename), label)

    return filename, file, label_file

def convert_tn3k(name='TN3K', target='thyroid_tumor'):

    converted_path = os.path.join(data_root(), 'Ultrasound', name, 'converted')
    ensure_directories(converted_path, [target])
    extracted_base = os.path.join(data_root(), 'Ultrasound', name, 'extracted', 'Thyroid Dataset', name.lower())

    filename_map = {}
    idx = 0

    if name == 'TN3K':
        split_map = {'train': 'trainval', 'test': 'test'}

        for split in ['train', 'test']:
            image_files = [os.path.join(extracted_base, f'{split_map[split]}-image', f) for f in os.listdir(os.path.join(extracted_base, f'{split_map[split]}-image'))]

            futures = []
            with ThreadPoolExecutor() as executor:
                for file in image_files:
                    futures.append(executor.submit(process_image, file, converted_path, target, idx))
                    idx += 1

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {name} {split} images"):
                    filename, file, label_file = future.result()
                    filename_map[filename] = {
                        'image': file,
                        target: label_file
                    }
                    split_map[split] = filename
    else:
        image_files = [os.path.join(extracted_base, 'thyroid-image', f) for f in os.listdir(os.path.join(extracted_base, 'thyroid-image'))]

        futures = []
        with ThreadPoolExecutor() as executor:
            for file in image_files:
                futures.append(executor.submit(process_image, file, converted_path, target, idx))
                idx += 1

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {name} images"):
                filename, file, label_file = future.result()
                filename_map[filename] = {
                    'image': file,
                    target: label_file
                }

    if name == 'TN3K':
        save_mapping(converted_path, split_map, json_fn='split_map')
    save_mapping(converted_path, filename_map)
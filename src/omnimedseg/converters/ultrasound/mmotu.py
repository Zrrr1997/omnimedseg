from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, split, split_map, label_map, converted_path, idx):
    filename = f"{idx:06d}.png"

    image = cv2.imread(file)
    label_file = file.replace('/image/', label_map[split]).replace('/train_image/', label_map[split]).replace('JPG', 'PNG')

    label = cv2.imread(label_file)

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'ovarian_tumor', filename), label)

    return filename, file

def convert_mmotu():
    converted_path = os.path.join(data_root(), 'Ultrasound', 'MMOTU', 'converted')
    ensure_directories(converted_path, ['ovarian_tumor'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join(data_root(), 'Ultrasound', 'MMOTU', 'extracted', 'dataset', 'OTU_2D')

    split_map = {'train': 'train_', 'test': ''}
    split_dict = {'train': [], 'test': []}
    label_map = {'train': '/train_label/label/', 'test': '/label/black_write/'}

    for split in ['train', 'test']:
        image_files = [os.path.join(extracted_base, split, f'{split_map[split]}image', f) for f in os.listdir(os.path.join(extracted_base, split, f'{split_map[split]}image'))]

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_image, file, split, split_map, label_map, converted_path, idx + i): file for i, file in enumerate(image_files)}

            for future in tqdm(as_completed(futures), total=len(image_files), desc=f"Processing MMOTU {split} images"):
                filename, file = future.result()
                filename_map[filename] = {
                    'image': file,
                    'fetal_head': file.replace('image', 'label')
                }
                split_dict[split].append(filename)
                idx += 1

    save_mapping(converted_path, split_dict, json_fn='split_map')
    save_mapping(converted_path, filename_map)

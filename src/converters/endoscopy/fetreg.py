import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def process_image(args):
    file, idx, converted_path = args
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('/images/', '/labels/')
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    vessels = (label == 1).astype(np.uint8) * 255
    tool = (label == 2).astype(np.uint8) * 255
    fetus = (label == 3).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'placental_vessels', filename), vessels)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'fetus', filename), fetus)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'surgical_tool', filename), tool)

    return filename, file, label_file

def convert_fetreg():
    converted_path = os.path.join('datasets', 'Endoscopy', 'FETREG', 'converted')
    ensure_directories(converted_path, ['placental_vessels', 'fetus', 'surgical_tool'])
    idx = 0
    filename_map = {}

    extracted_base = os.path.join('datasets', 'Endoscopy', 'FETREG', 'extracted', 'FetReg2021_dataset')

    tasks = []
    for split in ['Train', 'Test']:
        image_paths_base = os.path.join(extracted_base, split, f'{split}_FetReg2021_Task1_Segmentation')
        image_paths = [os.path.join(image_paths_base, f, 'images') for f in os.listdir(image_paths_base) if '.py' not in f]

        for image_path in image_paths:
            image_files = [os.path.join(image_path, f) for f in os.listdir(image_path)]
            for file in image_files:
                tasks.append((file, idx, converted_path))
                idx += 1

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing FETREG images"):
            filename, file, label_file = future.result()
            filename_map[filename] = {
                'image': file,
                **{k: label_file for k in ['placental_vessels', 'fetus', 'surgical_tool']}
            }

    save_mapping(converted_path, filename_map)
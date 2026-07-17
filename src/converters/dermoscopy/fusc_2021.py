
import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *

def convert_fusc_2021():
    converted_path = os.path.join('datasets', 'Dermoscopy', 'FUSC_2021', 'converted')
    ensure_directories(converted_path, ['foot_ulcer'])

    extracted_base = os.path.join('datasets', 'Dermoscopy', 'FUSC_2021', 'extracted', 'wound-segmentation-master', 'data', 'Foot Ulcer Segmentation Challenge')
    image_paths = []
    for split in ['train', 'validation']:
        image_paths += [os.path.join(extracted_base, split, 'images', f) for f in os.listdir(os.path.join(extracted_base, split, 'images'))]
    idx = 0

    filename_map = {}
    split_map = {'train': [], 'val': []}

    for file in tqdm(image_paths, desc="🖼️  Processing FUSC_2021 images"):
        filename = f"{idx:06d}.png"

        image = cv2.imread(file)
        label_file = file.replace('images', 'labels')
        label = cv2.imread(label_file)

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'foot_ulcer', filename), label)


        filename_map[filename] = {
            'image': file,
            'foot_ulcer': label_file
        }
        if 'train' in file:
            split_map['train'].append(filename)
        else:
            split_map['val'].append(filename)
        idx += 1
    save_mapping(converted_path, filename_map)
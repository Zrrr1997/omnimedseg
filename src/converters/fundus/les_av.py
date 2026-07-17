import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *

def convert_les_av():
    # Define paths
    converted_path = os.path.join('datasets', 'Fundus', 'LES_AV', 'converted')
    ensure_directories(converted_path, ['vessels', 'arteries', 'veins'])

    extracted_base = os.path.join('datasets', 'Fundus', 'LES_AV', 'extracted', 'LES-AV')

    idx = 0
    filename_map = {}

    # Define how extracted labels are mapped
    labels = ['image', 'vessels', 'arteries', 'veins']
    folder_names_extracted = {
        'image': 'images',
        'vessels': 'vessel-segmentations',
        'arteries': 'arteries',
        'veins': 'veins',
    }

    paths_extracted = {
        label: os.path.join(extracted_base, folder_names_extracted[label]) for label in labels
    }

    paths_converted = {
        'image': os.path.join(converted_path, 'images'),
        **{label: os.path.join(converted_path, 'labels', label) for label in labels if label != 'image'}
    }

    image_dir = paths_extracted['image']
    for file in tqdm(os.listdir(image_dir), desc="🖼️ Processing LES_AV images"):
        filename = f"{idx:06d}.png"
        filename_map[filename] = {
            label: os.path.join(paths_extracted[label], file) for label in labels
        }

        for label in labels:
            src_path = os.path.join(paths_extracted[label], file)
            dst_path = os.path.join(paths_converted[label], filename)
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)

        idx += 1

    # Save mapping
    save_mapping(converted_path, filename_map)
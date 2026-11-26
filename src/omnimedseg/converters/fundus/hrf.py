import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_hrf():
    # Define paths
    converted_path = os.path.join('datasets', 'Fundus', 'HRF', 'converted')
    ensure_directories(converted_path, ['vessels'])

    extracted_base = os.path.join('datasets', 'Fundus', 'HRF', 'extracted')
    convention_map = {'image': 'images', 'vessels': 'manual1'}

    idx = 0
    filename_map = {}

    for file in tqdm(os.listdir(os.path.join(extracted_base, convention_map['image'])), desc="🖼️  Processing images..."):
        image_file = os.path.join(extracted_base, convention_map['image'], file)
        label_file = image_file.replace(convention_map['image'], convention_map['vessels'])
        label_file = label_file.replace('.jpg', '.tif')

        filename = f"{idx:06d}.png"
        filename_map[filename] = {
            'image': image_file,
            'vessels': label_file
        }

        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            img = cv2.imread(image_file)
            if img is not None:
                cv2.imwrite(os.path.join(converted_path, 'images', filename), img)
            label = cv2.imread(label_file)
            if label is not None:
                cv2.imwrite(os.path.join(converted_path, 'labels','vessels', filename), label)

            idx += 1

    save_mapping(converted_path, filename_map)
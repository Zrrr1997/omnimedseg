import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_ravir():
    converted_path = os.path.join('datasets', 'Fundus', 'RAVIR', 'converted')
    ensure_directories(converted_path, ['vessels'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join('datasets', 'Fundus', 'RAVIR', 'extracted')
    image_files = [os.path.join(extracted_base, 'RAVIR Dataset', 'train', 'training_images', f) for f in os.listdir(os.path.join(extracted_base, 'RAVIR Dataset', 'train', 'training_images'))]

    for file in tqdm(sorted(image_files), desc="🖼️  Processing RAVIR images"):
        filename = f"{idx:06d}.png"
        shutil.copy(file, os.path.join(converted_path, 'images', filename))
        shutil.copy(file.replace('training_images', 'training_masks'), os.path.join(converted_path, 'labels', 'vessels', filename))

        filename_map[filename] = {'image': file, 'vessels': file.replace('training_images', 'training_masks')}
        idx += 1
    save_mapping(converted_path, filename_map)

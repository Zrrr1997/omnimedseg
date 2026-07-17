import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.utils.dataset_utils import *

def process_file(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('/images/', '/masks/')
    label = cv2.imread(label_file)

    image_path = os.path.join(converted_path, 'images', filename)
    label_path = os.path.join(converted_path, 'labels', 'polyp', filename)
    cv2.imwrite(image_path, image)
    cv2.imwrite(label_path, label)

    return filename, {'image': file, 'polyp': label_file}

def convert_hyper_kvasir():
    converted_path = os.path.join('datasets', 'Endoscopy', 'HYPER_KVASIR', 'converted')
    ensure_directories(converted_path, ['polyp'])
    extracted_base = os.path.join('datasets', 'Endoscopy', 'HYPER_KVASIR', 'extracted', 'segmented-images', 'images')
    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    filename_map = {}
    args_list = [(idx, file, converted_path) for idx, file in enumerate(image_files)]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, args_list), total=len(args_list), desc="🖼️  Processing HYPER_KVASIR images"))

    for filename, mapping in results:
        filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
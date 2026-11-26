from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_file(args):
    file, split, idx, converted_path = args
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    if split == 'train':
        label_file = file.replace('/Raw/', '/Masks/').replace('_raw.png', '_class.png')
    else:
        label_file = file.replace('Testing/OP', 'OP').replace('_raw.png', '_class.png')
    label = cv2.imread(label_file)
    label_shaft = ((label == 160).all(axis=2)).astype(np.uint8) * 255
    label_manipulator = ((label == 70).all(axis=2)).astype(np.uint8) * 255
    label_instrument = ((label > 0).all(axis=2)).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'shaft', filename), label_shaft)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'manipulator', filename), label_manipulator)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'instrument', filename), label_instrument)

    return filename, file, label_file

def convert_endovis_15():
    converted_path = os.path.join(data_root(), 'Endoscopy', 'ENDOVIS_15', 'converted')
    ensure_directories(converted_path, ['shaft', 'manipulator', 'instrument'])
    idx = 0
    filename_map = {}
    split_map = {'train': [], 'test': []}
    split_dict = {'train': 'Segmentation_Rigid_Training/Training', 'test': 'Testing'}
    split_subpath = {'train': '/Raw/', 'test': ''}

    extracted_base = os.path.join(data_root(), 'Endoscopy', 'ENDOVIS_15', 'extracted')

    for split in ['train', 'test']:
        image_paths = [os.path.join(extracted_base, split_dict[split], f) for f in os.listdir(os.path.join(extracted_base, split_dict[split])) if 'OP' in f]
        image_files = []
        for image_path in image_paths:
            image_files += [os.path.join(image_path + split_subpath[split], f) for f in os.listdir(image_path + split_subpath[split])]

        args_list = [(file, split, idx + i, converted_path) for i, file in enumerate(image_files)]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"🖼️  Processing ENDOVIS_15 {split} images"):
                filename, file, label_file = future.result()
                split_map[split].append(filename)
                filename_map[filename] = {
                    'image': file,
                    **{k: label_file for k in ['shaft', 'manipulator', 'instrument']}
                }
        idx += len(image_files)

    save_mapping(converted_path, split_map, 'split_map')
    save_mapping(converted_path, filename_map)
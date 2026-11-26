from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *
import threading

def convert_amd_sd():
    converted_path = os.path.join(data_root(), 'OCT', 'AMD_SD', 'converted')
    ensure_directories(converted_path, ['intraretinal_fluid', 'subretinal_fluid', 'PED', 'SHRM', 'IS_OS'])

    extracted_base = os.path.join(data_root(), 'OCT', 'AMD_SD', 'extracted', 'AMD-SD', 'images')

    with open(os.path.join(data_root(), 'OCT', 'AMD_SD', 'extracted', 'AMD-SD', 'training.txt'), 'r') as f:
        training_ids = set(el.strip() for el in f.readlines())

    with open(os.path.join(data_root(), 'OCT', 'AMD_SD', 'extracted', 'AMD-SD', 'validation.txt'), 'r') as f:
        validation_ids = set(el.strip() for el in f.readlines())

    image_files = []
    for i in np.arange(1, 157):
        dir_path = os.path.join(extracted_base, str(i))
        if not os.path.isdir(dir_path):
            continue
        image_files += [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

    filename_map = {}
    split_map = {'train': [], 'validation': []}
    idx_lock = threading.Lock()
    idx = [0]  # use list for mutability in threads

    # Class color map
    class_colors = {
        0: np.array([0, 0, 0]),
        1: np.array([255, 0, 0]),      # SRF
        2: np.array([0, 255, 0]),      # IRF
        3: np.array([0, 0, 255]),      # PED
        4: np.array([255, 27, 203]),   # IS_OS
        5: np.array([255, 255, 0])     # SHRM
    }
    color_values = np.array(list(class_colors.values()))

    def process(file):
        image_label = cv2.imread(file)
        if image_label is None:
            return None  # skip corrupted files

        half_width = image_label.shape[1] // 2
        image = image_label[:, :half_width, :]
        label = cv2.cvtColor(image_label[:, half_width:], cv2.COLOR_BGR2RGB)

        # Classify each pixel by nearest color
        flat_img = label.reshape(-1, 3)
        dists = np.linalg.norm(flat_img[:, None, :] - color_values[None, :, :], axis=2)
        class_indices = np.argmin(dists, axis=1)
        label_map = class_indices.reshape(label.shape[:2]).astype(np.uint8)

        # Create binary masks
        masks = {
            'subretinal_fluid': (label_map == 1).astype(np.uint8) * 255,
            'intraretinal_fluid': (label_map == 2).astype(np.uint8) * 255,
            'PED': (label_map == 3).astype(np.uint8) * 255,
            'IS_OS': (label_map == 4).astype(np.uint8) * 255,
            'SHRM': (label_map == 5).astype(np.uint8) * 255
        }

        # Thread-safe index assignment
        with idx_lock:
            local_idx = idx[0]
            idx[0] += 1
            filename = f"{local_idx:06d}.png"

        # Save images and masks
        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        for label_name, mask in masks.items():
            cv2.imwrite(os.path.join(converted_path, 'labels', label_name, filename), mask)

        # Determine split
        basename = os.path.basename(file)
        if basename in training_ids:
            split = 'train'
        elif basename in validation_ids:
            split = 'validation'
        else:
            print(f'❌ Image {file} not found in train/val lists!')
            split = None

        return filename, file, split

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process, file) for file in image_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing AMD_SD images"):
            result = future.result()
            if result is None:
                continue
            filename, file, split = result
            filename_map[filename] = {
                'image': file,
                **{k: file for k in ['intraretinal_fluid', 'subretinal_fluid', 'PED', 'SHRM', 'IS_OS']}
            }
            if split:
                split_map[split].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')
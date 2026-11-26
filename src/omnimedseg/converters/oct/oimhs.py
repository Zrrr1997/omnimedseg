import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *
import threading

def convert_oimhs():
    converted_path = os.path.join('datasets', 'OCT', 'OIMHS', 'converted')
    ensure_directories(converted_path, ['macular_hole', 'intraretinal_fluid', 'retina', 'choroid'])

    extracted_base = os.path.join('datasets', 'OCT', 'OIMHS', 'extracted', 'Images')

    image_files = []
    for i in np.arange(1, 126):
        dir_path = os.path.join(extracted_base, str(i))
        if not os.path.isdir(dir_path):
            continue
        image_files += [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

    filename_map = {}
    idx_lock = threading.Lock()
    idx = [0]  # use list for mutability in threads

    # Class color map
    class_colors = {
        0: np.array([0, 0, 0]),
        1: np.array([255, 0, 0]),      # macular hole
        2: np.array([255, 255, 0]),      # choroid
        3: np.array([0, 255, 0]),      # retina
        4: np.array([0, 0, 255]),   # intraretinal cyst
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
            'macular_hole': (label_map == 1).astype(np.uint8) * 255,
            'choroid': (label_map == 2).astype(np.uint8) * 255,
            'retina': (label_map == 3).astype(np.uint8) * 255,
            'intraretinal_fluid': (label_map == 4).astype(np.uint8) * 255,
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


        return filename, file

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process, file) for file in image_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing OIMHS images"):
            result = future.result()
            if result is None:
                continue
            filename, file = result
            filename_map[filename] = {
                'image': file,
                **{k: file for k in ['macular_hole', 'intraretinal_fluid', 'retina', 'choroid']}
            }


    save_mapping(converted_path, filename_map)
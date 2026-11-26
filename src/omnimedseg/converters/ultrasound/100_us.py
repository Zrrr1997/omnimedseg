
from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def convert_100_us():
    converted_path = os.path.join(data_root(), 'Ultrasound', '100_US', 'converted')
    ensure_directories(converted_path, ['pancreatic_tumor'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join(data_root(), 'Ultrasound', '100_US', 'extracted')

    image_files = [os.path.join(extracted_base, '105 US Images', f) for f in os.listdir(os.path.join(extracted_base, '105 US Images')) if '.png' in f]

    for file in tqdm(image_files, desc="Processing 100_US images"):
        filename = f"{idx:06d}.png"

        image = cv2.imread(file)
        label_file = file.replace('Images', 'Masks').replace('.png', ' G man.png')
        for n in np.arange(81, 100):
            label_file = label_file.replace(f'0{n}', f'{n}')
        label = cv2.imread(label_file)



        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'pancreatic_tumor', filename), label)

        filename_map[filename] = {
            'image': file,
            'pancreatic_tumor': file.replace('Images', 'Masks')
        }
        idx += 1
    save_mapping(converted_path, filename_map)
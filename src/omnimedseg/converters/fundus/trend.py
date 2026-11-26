import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_trend():
    converted_path = os.path.join('datasets', 'Fundus', 'TREND', 'converted')
    ensure_directories(converted_path, ['vessels'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join('datasets', 'Fundus', 'TREND', 'extracted')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(os.path.join(extracted_base)) if ('1_HEALTHY' in f or '4_HEALTHY' in f) and '.zip' not in f]

    for path in image_paths:
        image_files = [os.path.join(path, el) for el in os.listdir(path)]
        for image_file in tqdm(sorted(image_files), desc="🖼️  Processing TREND images"):
            filename = f"{idx:06d}.png"
            img = cv2.imread(image_file)
            label_file = image_file.replace('1_HEALTHY YOUNG_RAW_GOOD_QUALITY', '2_HEALTHY YOUNG SEGMENTED')
            label_file = label_file.replace('4_HEALTHY OLD RAW_GOOD QUALITY', '5_HEALTHY OLD SEGMENTED')
            label_file = label_file.replace('.tif', '_SEG.png')
            label = cv2.imread(label_file)

            cv2.imwrite(os.path.join(converted_path, 'images', filename), img)
            cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels', filename), label)
            filename_map[filename] = {'image': image_file, 'vessels': label_file}
            idx += 1
    save_mapping(converted_path, filename_map)
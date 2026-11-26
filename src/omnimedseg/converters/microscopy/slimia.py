import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def convert_slimia():
    converted_path = os.path.join('datasets', 'Microscopy', 'SLIMIA', 'converted')
    ensure_directories(converted_path, ['spheroid'])
    extracted_base = os.path.join('datasets', 'Microscopy', 'SLIMIA', 'extracted')

    image_files = []
    for root, dirs, files in os.walk(extracted_base):
        if 'Images' in root:
            for file in files:
                if file.lower().endswith('.tiff'):
                    image_files.append(os.path.join(root, file))
    label_files = []
    for root, dirs, files in os.walk(extracted_base):
        if 'Manual segmentations' in root:
            for file in files:
                if file.lower().endswith('.tiff'):
                    label_files.append(os.path.join(root, file))

    filename_map = {}
    idx_map = {file: f"{i:06d}.png" for i, file in enumerate(image_files)}

    def process_file(file):
        idx_filename = idx_map[file]

        try:
            image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            label_file = file.replace('/Images/', '/Manual segmentations/') \
                             .replace('.ome', '') 
            for n in ['500', '1000', '1500', '2000', '3000', '4000']:
                 label_file = label_file.replace(f'OVCAR8/{n}_Cells/', f'OVCAR8/{n}_cells/')
            label_file_orig = label_file 
            if not os.path.exists(label_file):
                label_file = label_file.replace('X.tiff', 'x.tiff')
            if not os.path.exists(label_file):
                label_file = label_file.replace('x.tiff', 'X.tiff')
            if not os.path.exists(label_file):
                label_file = label_file_orig.replace('.tiff', '.ome.tiff')
            if not os.path.exists(label_file):
                label_file = label_file_orig.replace('_B', 'h_B').replace('.ome', '')
                label_file = label_file.replace('hh', '')

            if not os.path.exists(label_file):
                return idx_filename, None, file, label_file  # Will skip
            label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)

            image_path = os.path.join(converted_path, 'images', idx_filename)
            label_path = os.path.join(converted_path, 'labels', 'spheroid', idx_filename)

            cv2.imwrite(image_path, image)
            cv2.imwrite(label_path, label)

            return idx_filename, {
                'image': file,
                'spheroid': label_file,
            }, None, None
        except Exception as e:
            return idx_filename, None, f"Error: {file}", str(e)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in image_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing SLIMIA images"):
            idx_filename, entry, missing_file, error = future.result()
            if entry:
                filename_map[idx_filename] = entry


    save_mapping(converted_path, filename_map)
    with open(os.path.join(converted_path, 'filename_map.json'), 'r') as f:
        json_file = json.load(f)
    saved_label_files = [v['spheroid'] for v in json_file.values()]
    for l in label_files:
        if l not in saved_label_files:
            print('Missing', l)
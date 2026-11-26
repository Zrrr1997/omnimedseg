from omnimedseg.core.paths import data_root
import os
import cv2
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_file(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.png"

    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    synapse = cv2.imread(file.replace('raw', 'synapses').replace('.tif', '.png'))
    mitochondria = cv2.imread(file.replace('raw', 'mitochondria').replace('.tif', '.png'))
    membrane = cv2.imread(file.replace('raw', 'membranes').replace('.tif', '.png'))

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'synapse', filename), synapse)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'mitochondria', filename), mitochondria)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'membrane', filename), membrane)

    return filename, {
        'image': file,
        'synapse': file.replace('raw', 'synapses').replace('.tif', '.png'),
        'mitochondria': file.replace('raw', 'mitochondria').replace('.tif', '.png'),
        'membrane': file.replace('raw', 'membranes').replace('.tif', '.png')
    }

def convert_sstem():
    converted_path = os.path.join(data_root(), 'Microscopy', 'SSTEM', 'converted')
    ensure_directories(converted_path, ['synapse', 'mitochondria', 'membrane'])

    extracted_base = os.path.join(data_root(), 'Microscopy', 'SSTEM', 'extracted', 'groundtruth-drosophila-vnc-master', 'stack1', 'raw')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    filename_map = {}
    args_list = [(idx, file, converted_path) for idx, file in enumerate(image_paths)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing SSTEM images"):
            filename, mapping = f.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
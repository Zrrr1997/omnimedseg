from omnimedseg.core.paths import data_root
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from omnimedseg.utils.dataset_utils import *

def process_file(args):
    file, idx, converted_path = args
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('/ds/', '/CVC-ClinicDB/').replace('/img/', '/Ground Truth/').replace('.png', '.tif')
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    label = (label > 0).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'polyp', filename), label)

    return filename, file, label_file

def convert_cvc_clinicdb():
    converted_path = os.path.join(data_root(), 'Endoscopy', 'CVC_CLINICDB', 'converted')
    ensure_directories(converted_path, ['polyp'])
    extracted_base = os.path.join(data_root(), 'Endoscopy', 'CVC_CLINICDB', 'extracted', 'ds', 'img')
    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]
    filename_map = {}

    args_list = [(file, idx, converted_path) for idx, file in enumerate(image_files)]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, args_list), total=len(args_list), desc="Processing CVC_CLINICDB images"))

    for filename, file, label_file in results:
        filename_map[filename] = {
            'image': file,
            'polyp': label_file,
        }
    save_mapping(converted_path, filename_map)
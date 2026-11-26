
from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_dermoscopy import *
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_nevus():
    converted_path = os.path.join(data_root(), 'Dermoscopy', 'NEVUS', 'converted')
    ensure_directories(converted_path, ['nevus'])

    extracted_base = os.path.join(data_root(), 'Dermoscopy', 'NEVUS', 'extracted', 'ds', 'img')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]
    class_mapping = load_class_mapping(os.path.join(data_root(), 'Dermoscopy', 'NEVUS', 'extracted', 'meta.json'))

    filename_map = {}

    # Prepare arguments for parallel processing
    args_list = [(idx, file, converted_path, class_mapping, 'nevus') for idx, file in enumerate(image_paths)]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_ham_10000_nevus, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing NEVUS images"):
            result = future.result()
            if result:
                filename, image_file, label_file = result
                filename_map[filename] = {
                    'image': image_file,
                    'nevus': label_file
                }

    save_mapping(converted_path, filename_map)
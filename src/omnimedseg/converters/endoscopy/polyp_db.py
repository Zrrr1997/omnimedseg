import os 
import cv2
import threading
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_polyp_db():
    converted_path = os.path.join('datasets', 'Endoscopy', 'POLYP_DB', 'converted')
    ensure_directories(converted_path, ['polyp'])

    extracted_base = os.path.join(
        'datasets', 'Endoscopy', 'POLYP_DB', 'extracted', 'PolypDB', 'PolypDB_modality_wise'
    )
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    filename_map = {}
    idx = 0
    idx_lock = threading.Lock()

    def process_file(file):
        nonlocal idx
        with idx_lock:
            local_idx = idx
            idx += 1

        filename = f"{local_idx:06d}.png"
        image = cv2.imread(file)

        # Determine label file path
        label_file = file.replace('/images/', '/masks/').replace('.jpg', '.png')
        if not os.path.exists(label_file):
            label_file = label_file.replace('.png', '.jpg')

        label = cv2.imread(label_file)

        # Save processed files
        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'polyp', filename), label)

        return filename, file, label_file

    all_image_files = []
    for image_path in image_paths:
        image_dir = os.path.join(image_path, 'images')
        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        all_image_files.extend(image_files)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_file, file) for file in all_image_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing POLYP_DB images"):
            filename, file, label_file = future.result()
            filename_map[filename] = {
                'image': file,
                'polyp': label_file
            }

    save_mapping(converted_path, filename_map)
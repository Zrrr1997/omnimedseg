from omnimedseg.core.paths import data_root
import os 
import cv2
import threading
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_bagls():
    converted_path = os.path.join(data_root(), 'Endoscopy', 'BAGLS', 'converted')
    ensure_directories(converted_path, ['glottal_space'])
    filename_map = {}
    split_map = {'train': [], 'test': []}

    extracted_base = os.path.join(data_root(), 'Endoscopy', 'BAGLS', 'extracted')
    split_convention = {'train': 'training', 'test': 'test'}
    
    all_jobs = []
    idx = 0
    idx_lock = threading.Lock()  # Ensure unique filenames when indexing

    def process_file(file, split):
        nonlocal idx
        # Locking ensures idx is unique
        with idx_lock:
            local_idx = idx
            idx += 1

        filename = f"{local_idx:06d}.png"
        image = cv2.imread(file)
        label_file = file.replace('.png', '_seg.png')
        label = cv2.imread(label_file)

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'glottal_space', filename), label)

        return filename, file, label_file

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        for split in ['train', 'test']:
            split_dir = os.path.join(extracted_base, split_convention[split])
            image_files = [
                os.path.join(split_dir, f)
                for f in os.listdir(split_dir)
                if '_seg' not in f and '.meta' not in f
            ]

            for file in image_files:
                futures.append(executor.submit(process_file, file, split))

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing BAGLS images"):
            filename, file, label_file = future.result()
            filename_map[filename] = {
                'image': file,
                'glottal_space': label_file
            }
            split = 'train' if 'training' in file else 'test'
            split_map[split].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')
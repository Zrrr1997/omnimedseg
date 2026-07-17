import os
import cv2
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def convert_ccagt():
    converted_path = os.path.join('datasets', 'Microscopy', 'CCAGT', 'converted')
    label_names = {
        1: 'nucleus',
        2: 'cluster',
        3: 'satellite',
        4: 'nucleus_out_of_focus',
        5: 'overlapped_nuclei',
        6: 'non_viable_nucleus',
        7: 'white_blood_cell'
    }
    all_labels = list(label_names.values())
    ensure_directories(converted_path, all_labels + ['images'])

    extracted_base = os.path.join('datasets', 'Microscopy', 'CCAGT', 'extracted', 'wg4bpm33hj-2', 'images')
    extract_archives(extracted_base)
    extract_archives(extracted_base.replace('images', 'masks'))

    # Collect all image files
    image_dirs = [
        os.path.join(extracted_base, f)
        for f in os.listdir(extracted_base)
        if '.zip' not in f and os.path.isdir(os.path.join(extracted_base, f))
    ]

    all_image_files = [
        os.path.join(dir_path, f)
        for dir_path in image_dirs
        for f in os.listdir(dir_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    filename_map = {}
    lock = Lock()  # For thread-safe updates

    def process_image(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.png"
        label_path = file.replace('images', 'masks').replace('.jpg', '.png')
        image = cv2.imread(file)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if image is None or label is None:
            return None

        label_ids = np.unique(label)
        label_ids = label_ids[label_ids > 0]

        if len(label_ids) == 0:
            return None  # Skip if label is empty

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)

        mapping = {'image': file}
        for label_id in label_ids:
            if label_id not in label_names:
                continue
            binary_mask = (label == label_id).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(converted_path, 'labels', label_names[label_id], filename), binary_mask)
            mapping[label_names[label_id]] = label_path

        return filename, mapping

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(process_image, (idx, file)) for idx, file in enumerate(all_image_files)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing CCAGT images"):
            result = future.result()
            if result:
                filename, mapping = result
                with lock:
                    filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
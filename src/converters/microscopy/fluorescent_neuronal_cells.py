import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def process_image_fluorescent_neuronal_cells(file, idx, converted_path):
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('images', 'ground_truths/masks')
    label = cv2.imread(label_file)

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'neuronal_cell', filename), label)

    return filename, file, label_file

def convert_fluorescent_neuronal_cells():
    converted_path = os.path.join('datasets', 'Microscopy', 'FLUORESCENT_NEURONAL_CELLS', 'converted')
    ensure_directories(converted_path, ['neuronal_cell'])

    filename_map = {}
    split_map = {'trainval': [], 'test': []}
    extracted_base = os.path.join('datasets', 'Microscopy', 'FLUORESCENT_NEURONAL_CELLS', 'extracted')

    idx = 0
    futures = []
    tasks = []

    for color in ['red', 'green', 'yellow']:
        for split in ['trainval', 'test']:
            image_dir = os.path.join(extracted_base, color, split, 'images')
            image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

            for file in image_paths:
                tasks.append((file, idx, split))
                idx += 1

    with ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(process_image_fluorescent_neuronal_cells, file, i, converted_path): (i, split)
                          for file, i, split in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="🖼️  Processing FLUORESCENT_NEURONAL_CELLS images"):
            i, split = future_to_task[future]
            try:
                filename, image_path, label_path = future.result()
                filename_map[filename] = {
                    'image': image_path,
                    'neuronal_cell': label_path
                }
                split_map[split].append(filename)
            except Exception as e:
                print(f"Error processing image index {i}: {e}")

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')
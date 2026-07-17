import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def process_file(file, idx, converted_path):
    filename = f"{idx:06d}.png"

    image_label = cv2.imread(file)

    blue, green, red = cv2.split(image_label)

    # Convert to int16 to avoid overflow/underflow in subtraction
    red = red.astype(np.int16)
    green = green.astype(np.int16)
    blue = blue.astype(np.int16)

    # Compute redness score: red minus the average of green and blue
    redness = red - ((green + blue) // 2)

    # Threshold the redness to get a binary mask (label)
    label = (redness > 30).astype(np.uint8) * 255  # You can adjust the threshold

    image = blue.astype(np.uint8)
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'lesion', filename), label)

    return filename, file

def convert_oct_lesion():
    converted_path = os.path.join('datasets', 'OCT', 'OCT_LESION', 'converted')
    ensure_directories(converted_path, ['lesion'])

    extracted_base = os.path.join('datasets', 'OCT', 'OCT_LESION', 'extracted')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, idx, converted_path): idx for idx, file in enumerate(image_files)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing OCT_LESION images"):
            filename, file = future.result()
            filename_map[filename] = {
                'image': file,
                'lesion': file,
            }

    save_mapping(converted_path, filename_map)
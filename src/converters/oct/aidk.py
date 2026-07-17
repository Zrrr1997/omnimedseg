import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def convert_single_image_aidk(idx, file, converted_path, filename_map_lock):
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('.bmp', '.json').replace('Original_AS-OCT_Images', 'Experts_Annotations')

    with open(label_file, 'r') as f:
        label_data = json.load(f)
        contours = label_data['shapes']
        contour_masks = {}
        for contour in contours:
            contour_mask = contour_to_mask(
                np.array(contour['points']),
                (label_data['imageHeight'], label_data['imageWidth'], 3)
            ).astype(np.uint8) * 255

            label = contour['label'].lower()

            if label not in contour_masks:
                contour_masks[label] = contour_mask
            else:
                contour_masks[label] += contour_mask

        if 'lesion' not in contour_masks:
            contour_masks['lesion'] = np.zeros(image.shape[:-1], dtype=np.uint8)
        if 'iris' not in contour_masks:
            contour_masks['iris'] = np.zeros(image.shape[:-1], dtype=np.uint8)

    # Save images
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'cornea', filename), contour_masks['cornea'])
    cv2.imwrite(os.path.join(converted_path, 'labels', 'iris', filename), contour_masks['iris'])
    cv2.imwrite(os.path.join(converted_path, 'labels', 'lesion', filename), contour_masks['lesion'])

    return filename, {
        'image': file,
        **{k: label_file for k in ['cornea', 'lesion', 'iris']}
    }

def convert_aidk():
    converted_path = os.path.join('datasets', 'OCT', 'AIDK', 'converted')
    ensure_directories(converted_path, ['cornea', 'lesion', 'iris', 'images'])

    extracted_base = os.path.join('datasets', 'OCT', 'AIDK', 'extracted', 'AIDK_Dataset')

    image_files = []
    image_files += [
        os.path.join(extracted_base, 'Full-frame_Dataset', 'Original_AS-OCT_Images', f)
        for f in os.listdir(os.path.join(extracted_base, 'Full-frame_Dataset', 'Original_AS-OCT_Images'))
        if f.endswith('.bmp')
    ]
    image_files += [
        os.path.join(extracted_base, 'Partial-frame_Dataset', 'Original_AS-OCT_Images', f)
        for f in os.listdir(os.path.join(extracted_base, 'Partial-frame_Dataset', 'Original_AS-OCT_Images'))
        if f.endswith('.bmp')
    ]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(convert_single_image_aidk, idx, file, converted_path, None): idx
            for idx, file in enumerate(image_files)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing AIDK images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
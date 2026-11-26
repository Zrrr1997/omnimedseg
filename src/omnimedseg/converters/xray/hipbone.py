import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(file, converted_path):
    try:
        filename = os.path.basename(file).replace('.nii.gz', '.png')

        image = nib.load(file).get_fdata().astype(np.uint8)
        label_file = file.replace('image', 'label')
        label = nib.load(label_file).get_fdata()
        femur = (label == 1).astype(np.uint8) * 255
        hipbone = (label == 2).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'hipbone', filename), hipbone)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'femur', filename), femur)

        return filename, {
            'image': file,
            'femur': label_file,
            'hipbone': label_file
        }
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def convert_hipbone():
    converted_path = os.path.join('datasets', 'XRay', 'HIPBONE', 'converted')
    ensure_directories(converted_path, ['hipbone', 'femur'])

    extracted_base = os.path.join('datasets', 'XRay', 'HIPBONE', 'extracted', 'images')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, file, converted_path): file for file in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing HIPBONE images"):
            result = future.result()
            if result:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
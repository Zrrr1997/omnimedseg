import os
import cv2
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_file(file, converted_path):
    try:
        filename = f"{os.path.basename(file).split('.')[0]}.png"

        image = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image)[1]

        label = sitk.ReadImage(file.replace('image_mha', 'label_mha'))
        label = sitk.GetArrayFromImage(label)

        label_ps = (label == 1).astype(np.uint8) * 255
        label_fh = (label == 2).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'pubic_symphysis', filename), label_ps)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'fetal_head', filename), label_fh)

        return filename, {
            'image': file,
            'pubic_symphysis': file.replace('image_mha', 'label_mha'),
            'fetal_head': file.replace('image_mha', 'label_mha')
        }
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def convert_psfhs():
    converted_path = os.path.join('datasets', 'Ultrasound', 'PSFHS', 'converted')
    ensure_directories(converted_path, ['pubic_symphysis', 'fetal_head'])
    filename_map = {}

    extracted_base = os.path.join('datasets', 'Ultrasound', 'PSFHS', 'extracted', 'Pubic Symphysis-Fetal Head Segmentation and Angle of Progression')
    image_files = [os.path.join(extracted_base, 'image_mha', f) for f in os.listdir(os.path.join(extracted_base, 'image_mha')) if '.mha' in f]

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, converted_path): file for file in image_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing PSFHS images"):
            result = future.result()
            if result:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
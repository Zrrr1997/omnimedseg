import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def process_image(args):
    idx, file, converted_path = args
    import nibabel as nib
    import numpy as np

    filename = f"{idx:06d}.png"
    image = (nib.load(file).get_fdata()[:,:,0]* 255).astype(np.uint8) 

    label_file = file.replace('images', 'labels').replace('_0000', '').replace('_0001', '').replace('_0002', '')
    label = nib.load(label_file).get_fdata()[:,:,0]

    result = {}
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    if 'Task015_RPE1' in file:
        cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus', filename), label.astype(np.uint8) * 255)
        result = {
            'filename': filename,
            'mapping': {
                'image': file,
                'nucleus': label_file,
            }
        }
    else:
        nucleus_center = (label == 1).astype(np.uint8) * 255
        nucleus_border = (label == 2).astype(np.uint8) * 255
        nucleus = (label > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus', filename), nucleus)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus_center', filename), nucleus_center)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus_border', filename), nucleus_border)
        result = {
            'filename': filename,
            'mapping': {
                'image': file,
                **{k: label_file for k in ['nucleus', 'nucleus_border', 'nucleus_center']}
            }
        }
    return result

def convert_brifiseg():
    converted_path = os.path.join('datasets', 'Microscopy', 'BRIFISEG', 'converted')
    ensure_directories(converted_path, ['nucleus', 'nucleus_border', 'nucleus_center'])

    extracted_base = os.path.join('datasets', 'Microscopy', 'BRIFISEG', 'extracted')
    image_files = []
    image_files += [os.path.join(extracted_base, 'Task015_RPE1', 'imagesTr', f) for f in os.listdir(os.path.join(extracted_base, 'Task015_RPE1', 'imagesTr'))]
    image_files += [os.path.join(extracted_base, 'Task015_RPE1', 'imagesTs', f) for f in os.listdir(os.path.join(extracted_base, 'Task015_RPE1', 'imagesTs'))]
    image_files += [os.path.join(extracted_base, 'Task019_A549', 'imagesTr', f) for f in os.listdir(os.path.join(extracted_base, 'Task019_A549', 'imagesTr'))]
    image_files += [os.path.join(extracted_base, 'Task019_A549', 'imagesTs', f) for f in os.listdir(os.path.join(extracted_base, 'Task019_A549', 'imagesTs'))]

    filename_map = {}

    args_list = [(idx, file, converted_path) for idx, file in enumerate(image_files)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing BRIFISEG images"):
            result = f.result()
            filename_map[result['filename']] = result['mapping']

    save_mapping(converted_path, filename_map)

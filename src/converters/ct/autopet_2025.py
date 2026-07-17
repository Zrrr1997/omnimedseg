import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils.dataset_utils import *


def _process_autopet_2024_ct(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)

    # Copy label
    label_file = file.replace('imagesTr', 'labelsTr').replace('_0000', '')
    dst_label = os.path.join(converted_path, 'labels', 'lesion', filename)
    shutil.copy(label_file, dst_label)

    return filename, {
        'image': file,
        'lesion': label_file
    }


def convert_autopet_2024_ct(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'AUTOPET_2024_CT', 'converted')
    label_classes = ['lesion']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'AUTOPET_2024_CT', 'extracted')
    image_paths = [os.path.join(extracted_base, 'imagesTr', el) for el in os.listdir(os.path.join(extracted_base, 'imagesTr')) if '_0000' in el]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_autopet_2024_ct, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing AUTOPET_2024_CT images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

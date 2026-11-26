import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *


def _process_atm(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)

    # Copy label
    label_file = file.replace('imagesTr', 'labelsTr')
    dst_label = os.path.join(converted_path, 'labels', 'airways', filename)
    shutil.copy(label_file, dst_label)

    return filename, {
        'image': file,
        'airways': label_file
    }


def convert_atm(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'ATM', 'converted')
    label_classes = ['airways']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'ATM', 'extracted')
    image_paths = []
    image_paths += [os.path.join(extracted_base, 'TrainBatch1', 'imagesTr', f)
                    for f in os.listdir(os.path.join(extracted_base, 'TrainBatch1', 'imagesTr'))]
    image_paths += [os.path.join(extracted_base, 'TrainBatch2', 'imagesTr', f)
                    for f in os.listdir(os.path.join(extracted_base, 'TrainBatch2', 'imagesTr'))]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_atm, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing ATM images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

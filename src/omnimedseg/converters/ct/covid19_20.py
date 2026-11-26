from omnimedseg.core.paths import data_root
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *


def _process_covid19_20(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)

    # Copy label
    label_file = file.replace('_ct.nii.gz', '_seg.nii.gz')
    dst_label = os.path.join(converted_path, 'labels', 'lung_lesion', filename)
    shutil.copy(label_file, dst_label)

    return filename, {
        'image': file,
        'lung_lesion': label_file
    }


def convert_covid19_20(num_workers=8):
    converted_path = os.path.join(data_root(), 'CT', 'COVID19_20', 'converted')
    label_classes = ['lung_lesion']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'COVID19_20', 'extracted', 'COVID-19-20_v2', 'Train')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(os.path.join(extracted_base)) if '_ct.nii.gz' in f]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_covid19_20, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing COVID19_20 images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

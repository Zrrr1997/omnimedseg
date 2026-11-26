from omnimedseg.core.paths import data_root
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from omnimedseg.utils.postprocess_download import *

def _process_sts_tooth(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)

    # Copy label
    label_file = file.replace('Image', 'Mask')
    dst_label = os.path.join(converted_path, 'labels', 'tooth', filename)
    shutil.copy(label_file, dst_label)

    return filename, {
        'image': file,
        'tooth': label_file
    }


def convert_sts_tooth_ct(num_workers=8):
    converted_path = os.path.join(data_root(), 'CT', 'STS_TOOTH_CT', 'converted')
    label_classes = ['tooth']
    ensure_directories(converted_path, label_classes)
    extract_with_7z(os.path.join(data_root(), 'CT', 'STS_TOOTH_CT', 'SD-Tooth.zip.001'), os.path.join(data_root(), 'CT', 'STS_TOOTH_CT', 'extracted'))

    extracted_base = os.path.join(data_root(), 'CT', 'STS_TOOTH_CT', 'extracted', 'SD-Tooth', 'STS-3D-Tooth', 'Integrity', 'Labeled', 'Image')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(os.path.join(extracted_base)) if '_005' not in f and '_004' not in f] # Labels are just wrong for these two cases, so we skip them

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_sts_tooth, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing STS_TOOTH_CT images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

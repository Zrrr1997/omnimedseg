from omnimedseg.core.paths import data_root
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *


def _process_vessel12(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Convert image
    convert_mhd_raw_to_nifti(file, os.path.join(converted_path, 'images', filename))

    # Convert label
    label_file = re.sub(r'01-05|06-10|11_15|16-20', '01-20_Lungmasks', file)
    convert_mhd_raw_to_nifti(label_file, os.path.join(converted_path, 'labels', 'lungs', filename))

    return filename, {
        'image': [file, file.replace('mhd', 'raw')],
        'lungs': [label_file, label_file.replace('mhd', 'raw')],
    }


def convert_vessel12(num_workers=8):
    converted_path = os.path.join(data_root(), 'CT', 'VESSEL12', 'converted')
    label_classes = ['lungs']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'VESSEL12', 'extracted')
    subdirs = ['VESSEL12_01-05', 'VESSEL12_06-10', 'VESSEL12_11_15', 'VESSEL12_16-20']

    # Collect image paths
    image_paths = []
    for sd in subdirs:
        sd_path = os.path.join(extracted_base, sd)
        image_paths += [os.path.join(sd_path, f) for f in os.listdir(sd_path) if f.endswith('.mhd')]

    # Build args list
    args_list = [(i, file, converted_path) for i, file in enumerate(image_paths)]

    filename_map = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_vessel12, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing VESSEL12 images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

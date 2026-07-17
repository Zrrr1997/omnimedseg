import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.dataset_utils import *
from src.converters.utils_ct import *

def convert_pituitory_tumor():
    converted_path = os.path.join('datasets', 'MRI', 'PITUITORY_TUMOR', 'converted')

    # Create label mapping
    label_classes = ['carotid', 'pituitory_tumor']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'PITUITORY_TUMOR', 'extracted')

    image_paths = [os.path.join(extracted_base, el) for el in os.listdir(os.path.join(extracted_base)) if 'T1.nii' in el or 'T2_f.nii' in el]

    filename_map = {}
    print(
        "\n" + "=" * 80 +
        "\n⚠️  WARNING: --keep_raw is disabled for PITUITARY_TUMOR (too large)\n"
        "   To reproduce the raw data, run again with --extract\n"
        "   (make sure you have enough disk space!)\n" +
        "=" * 80 + "\n"
    )

    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        temp_dict = {'image': file}

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.move(file, dest_image_path)

        carotid = file.replace('T1', 'carotids_f').replace('T2_f', 'carotids_f') 
        if os.path.exists(carotid):
            shutil.copy(carotid, os.path.join(converted_path, 'labels', 'carotid', filename))
            temp_dict['carotid'] = carotid

        tumor = file.replace('T1', 'tumor_f').replace('T2_f', 'tumor_f') 
        if os.path.exists(tumor):
            shutil.copy(tumor, os.path.join(converted_path, 'labels', 'pituitory_tumor', filename))
            temp_dict['pituitory_tumor'] = tumor
        
        return filename, temp_dict


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing PITUITORY_TUMOR images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict



    save_mapping(converted_path, filename_map)

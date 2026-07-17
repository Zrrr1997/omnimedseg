import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils.dataset_utils import *
import SimpleITK as sitk

def _process_stacom_slawt_ct(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    img = sitk.ReadImage(file)
    dst_img = os.path.join(converted_path, 'images', filename)
    sitk.WriteImage(img, dst_img)

    # Copy label
    label_file = file.replace('.nii', '_la.nii')
    label = sitk.ReadImage(label_file)
    dst_label = os.path.join(converted_path, 'labels', 'left_atrium', filename)
    sitk.WriteImage(label, dst_label)

    return filename, {
        'image': file,
        'left_atrium': label_file
    }


def convert_stacom_slawt_ct(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'STACOM_SLAWT_CT', 'converted')
    label_classes = ['left_atrium']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'STACOM_SLAWT_CT', 'extracted')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(os.path.join(extracted_base)) if '.nii' in f and 'la' not in f]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_stacom_slawt_ct, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing STACOM_SLAWT_CT images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

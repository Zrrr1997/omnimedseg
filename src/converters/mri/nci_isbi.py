import os
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *
from src.converters.utils_ct import *


def process_image(idx, image_path, label_paths, converted_path):
    """Helper: process one image/label pair and return mapping info."""
    filename = f"{idx:06d}.nii.gz"

    # --- Load the image to get its affine ---
    image_im = nib.load(image_path)
    image_affine = image_im.affine

    # Copy the image to the converted folder
    shutil.copy(image_path, os.path.join(converted_path, 'images', filename))

    # --- Match corresponding label ---
    label_path = [el for el in label_paths if os.path.basename(os.path.dirname(image_path)) in el][0]
    label_im = convert_nrrd_nii(label_path)
    label_data = label_im.get_fdata()

    # --- Create binary masks (flipped along axis=1) ---
    cg_mask = np.flip((label_data == 2).astype('uint8'), axis=1)
    pz_mask = np.flip((label_data == 1).astype('uint8'), axis=1)

    # --- Use the image affine for both labels ---
    cg_img = nib.Nifti1Image(cg_mask, affine=image_affine)
    pz_img = nib.Nifti1Image(pz_mask, affine=image_affine)

    # --- Save ---
    nib.save(cg_img, os.path.join(converted_path, 'labels', 'prostate_central_gland', filename))
    nib.save(pz_img, os.path.join(converted_path, 'labels', 'prostate_peripheral_zone', filename))

    # Determine dataset split
    if 'Train' in image_path:
        split = 'train'
    elif 'Test' in image_path:
        split = 'test'
    else:
        split = 'leaderboard'

    return {
        'filename': filename,
        'split': split,
        'mapping': {
            'image': image_path,
            'prostate_central_gland': label_path,
            'prostate_peripheral_zone': label_path,
        },
    }


def convert_nci_isbi():
    converted_path = os.path.join('datasets', 'MRI', 'NCI_ISBI', 'converted')
    label_classes = ['prostate_central_gland', 'prostate_peripheral_zone']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'NCI_ISBI', 'extracted')
    compressed_base = os.path.join('datasets', 'MRI', 'NCI_ISBI', 'compressed')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('NCI_ISBI', extracted_base, converted_path)

    # --- Collect all image and label paths ---
    image_dirs = [os.path.join(converted_path, 'all_niftis', el)
                  for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]
    image_paths = [
        os.path.join(image_dir, [el for el in os.listdir(image_dir) if '.nii.gz' in el][0])
        for image_dir in image_dirs
    ]

    label_paths = []
    for subset in ['Test', 'Training', 'Leaderboard']:
        label_paths += [
            os.path.join(extracted_base, subset, el)
            for el in os.listdir(os.path.join(extracted_base, subset))
            if '.nrrd' in el
        ]

    filename_map = {}
    split_map = {'train': [], 'test': [], 'leaderboard': []}

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        futures = {
            executor.submit(process_image, idx, image_path, label_paths, converted_path): idx
            for idx, image_path in enumerate(image_paths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️ Converting NCI_ISBI..."):
            result = future.result()
            filename_map[result['filename']] = result['mapping']
            split_map[result['split']].append(result['filename'])

    # --- Save mappings ---
    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')



if __name__ == "__main__":
    convert_nci_isbi()

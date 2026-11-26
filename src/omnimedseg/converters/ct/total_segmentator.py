from omnimedseg.core.paths import data_root
import os
import shutil
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import nibabel as nib

import logging
logging.getLogger("nibabel").setLevel(logging.ERROR)  # only show real errors


def process_file_total_segmentator(idx_file_pair, extracted_base, converted_path, label_classes):
    idx, file = idx_file_pair
    filename = f"{idx:06d}.nii.gz"
    out_image_path = os.path.join(converted_path, 'images', filename)
    if os.path.exists(file):
        shutil.move(file, out_image_path)
    data_dict = {'image': file}
    dir_path = os.path.dirname(file)
    base_path = os.path.basename(os.path.dirname(file))
    label_files = [os.path.join(dir_path, 'segmentations', el) for el in os.listdir(os.path.join(dir_path, 'segmentations'))]
    for label_file in label_files:
        new_label_file = label_file
        if 'kidney_cyst' in new_label_file or 'rib_' in new_label_file:
            continue
        if '_left' in new_label_file:
            new_label_file = new_label_file.replace('_left', '').replace('segmentations/', 'segmentations/left_')
        if '_right' in new_label_file:
            new_label_file = new_label_file.replace('_right', '').replace('segmentations/', 'segmentations/right_')
        if '_hip' in new_label_file:
            new_label_file = new_label_file.replace('_hip', '_hipbone')
        if 'urinary_bladder' in new_label_file:
            new_label_file = new_label_file.replace('urinary_bladder', 'bladder')
        if 'small_bowel' in new_label_file:
            new_label_file = new_label_file.replace('small_bowel', 'small_intestine')
        if 'thyroid_gland' in new_label_file:
            new_label_file = new_label_file.replace('thyroid_gland', 'thyroid')
        if 'portal_vein_and_splenic_vein' in new_label_file:
            new_label_file = new_label_file.replace('portal_vein_and_splenic_vein', 'portal_vein_splenic_vein')
        if 'vertebrae' in new_label_file:
            new_label_file = new_label_file.replace('vertebrae_', '')
        new_label_file = new_label_file.replace('.nii.gz', '/' + filename).replace('segmentations', 'labels').replace('extracted', 'converted').replace(base_path, '').replace('/labels', 'labels')
        if os.path.exists(label_file):
            shutil.move(label_file, new_label_file)
        data_dict[os.path.basename(new_label_file).replace('.nii.gz', '')] = label_file
    # --- Kidney cysts ---
    left_kidney_cyst_path = os.path.join(dir_path, 'segmentations', 'kidney_cyst_left.nii.gz')
    right_kidney_cyst_path = os.path.join(dir_path, 'segmentations', 'kidney_cyst_right.nii.gz')

    left_img = nib.load(left_kidney_cyst_path)
    right_img = nib.load(right_kidney_cyst_path)

    # Combine voxel arrays
    left_data = left_img.get_fdata() > 0
    right_data = right_img.get_fdata() > 0
    combined_kidney_cyst = np.logical_or(left_data, right_data).astype(np.uint8)

    # Save merged kidney cyst
    kidney_cyst_img = nib.Nifti1Image(combined_kidney_cyst, affine=left_img.affine, header=left_img.header)
    out_kidney_cyst = os.path.join(converted_path, 'labels', 'kidney_cyst', filename)
    nib.save(kidney_cyst_img, out_kidney_cyst)
    data_dict['kidney_cyst'] = [left_kidney_cyst_path, right_kidney_cyst_path]

    # --- Ribs ---
    rib_label_files = [
        os.path.join(dir_path, 'segmentations', el)
        for el in os.listdir(os.path.join(dir_path, 'segmentations'))
        if 'rib_' in el
    ]
    data_dict['ribs'] = rib_label_files

    # Use the first rib as reference for affine/header
    ref_img = nib.load(rib_label_files[0])
    combined_ribs = np.zeros(ref_img.shape, dtype=np.uint8)

    for rib_file in rib_label_files:
        rib_img = nib.load(rib_file)
        rib_data = rib_img.get_fdata() > 0
        combined_ribs = np.logical_or(combined_ribs, rib_data)

    # Save merged ribs
    rib_img_out = nib.Nifti1Image(combined_ribs.astype(np.uint8), affine=ref_img.affine, header=ref_img.header)
    out_ribs = os.path.join(converted_path, 'labels', 'ribs', filename)
    nib.save(rib_img_out, out_ribs)

    return filename, data_dict



def convert_total_segmentator():
    print("note: --keep-raw has no effect for TOTAL_SEGMENTATOR (raw data is too large to keep); "
          "run `omnimedseg extract TOTAL_SEGMENTATOR` to recreate it")
    converted_path = os.path.join(data_root(), 'CT', 'TOTAL_SEGMENTATOR', 'converted')
    label_classes = [
        'brain', 'heart', 'liver', 'spleen', 'stomach', 'pancreas', 'gallbladder', 'bladder', 'kidney_cyst', 'left_kidney', 'right_kidney', 'spinal_cord',

        'left_lung_upper_lobe', 'left_lung_middle_lobe', 'left_lung_lower_lobe',
        'right_lung_upper_lobe', 'right_lung_middle_lobe', 'right_lung_lower_lobe',

        'left_adrenal_gland', 'right_adrenal_gland',

        'aorta', 'brachiocephalic_trunk', 'left_brachiocephalic_vein', 'right_brachiocephalic_vein',
        'left_common_carotid_artery', 'right_common_carotid_artery', 'left_iliac_artery', 'right_iliac_artery',
        'left_iliac_vena', 'right_iliac_vena', 'inferior_vena_cava', 'superior_vena_cava', 'portal_vein_splenic_vein',
        'pulmonary_vein', 'left_subclavian_artery', 'right_subclavian_artery',

        'left_autochthon', 'right_autochthon',
        'left_gluteus_maximus', 'right_gluteus_maximus',
        'left_gluteus_medius', 'right_gluteus_medius',
        'left_gluteus_minimus', 'right_gluteus_minimus',
        'left_iliopsoas', 'right_iliopsoas',

        'left_clavicula', 'right_clavicula',
        'left_humerus', 'right_humerus',
        'left_hipbone', 'right_hipbone',
        'left_femur', 'right_femur',
        'left_scapula', 'right_scapula',
        'ribs', 'sternum', 'sacrum', 'skull', 'costal_cartilages',

        'colon', 'duodenum', 'esophagus', 'small_intestine', 'trachea', 'thyroid', 'left_atrial_appendage', 'right_atrial_appendage', 'prostate', 'heart'
    ]    
    # Vertebrae
    for i in range(1, 13):
        label_classes.append(f'T{i}')
    for i in range(1, 6):
        label_classes.append(f'L{i}')
    for i in range(1, 8):
        label_classes.append(f'C{i}')
    label_classes.append('S1')
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'TOTAL_SEGMENTATOR', 'extracted')
    image_paths = []
    image_paths += sorted([
        os.path.join(extracted_base, f, 'ct.nii.gz')
        for f in os.listdir(os.path.join(extracted_base)) if '.csv' not in f and '.zip' not in f
    ])

    tasks = [(i, path) for i, path in enumerate(image_paths)]

    # Use partial to pass static args
    process_fn = partial(process_file_total_segmentator, extracted_base=extracted_base,
                         converted_path=converted_path, label_classes=label_classes)

    filename_map = {}


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_fn, task) for task in tasks]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TOTAL_SEGMENTATOR images"):
            results.append(future.result())

    for filename, mapping in results:
        filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

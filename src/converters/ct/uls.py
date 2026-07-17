import os
import shutil
from src.utils.dataset_utils import *
from src.converters.utils_ct import *
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file_uls(idx_file_pair, extracted_base, converted_path, label_classes):
    idx, file = idx_file_pair
    filename = f"{idx:06d}.nii.gz"
    out_image_path = os.path.join(converted_path, 'images', filename)
    if os.path.exists(file):
        shutil.move(file, out_image_path)

    label_file = file.replace('/ULS23/', '/ULS23-main/annotations/ULS23/').replace('/images/', '/labels/')
    out_label_path = os.path.join(converted_path, 'labels', 'lesion', filename)
    if os.path.exists(label_file):
        shutil.move(label_file, out_label_path)


    return filename, {
        'image': file,
        'label': label_file
    }

def convert_uls():
    print(
        "\n" + "=" * 80 +
        "\n⚠️  WARNING: --keep_raw is disabled for ULS (too large)\n"
        "   To reproduce the raw data, run again with --extract\n"
        "   (make sure you have enough disk space!)\n" +
        "=" * 80 + "\n"
    )
    converted_path = os.path.join('datasets', 'CT', 'ULS', 'converted')
    label_classes = ['lesion']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'ULS', 'extracted', 'ULS23')

    '''
    for subdir in ['novel_data/ULS23_DeepLesion3D', 'novel_data/ULS23_Radboudumc_Bone', 'novel_data/ULS23_Radboudumc_Pancreas', 'processed_data/fully_annotated/kits21', 'processed_data/fully_annotated/LIDC-IDRI', 'processed_data/fully_annotated/LiTS', 'processed_data/fully_annotated/MDSC_Task06_Lung', 'processed_data/fully_annotated/MDSC_Task07_Pancreas', 'processed_data/fully_annotated/MDSC_Task10_Colon', 'processed_data/fully_annotated/NIH_LN_ABD', 'processed_data/fully_annotated/NIH_LN_MED']:
        unzip_file(os.path.join(extracted_base, subdir, 'images.zip'), os.path.join(extracted_base, subdir))
    
    extracted_base_labels = os.path.join('datasets', 'CT', 'ULS', 'extracted', 'ULS23-main', 'annotations', 'ULS23')
    for subdir in ['novel_data/ULS23_DeepLesion3D', 'novel_data/ULS23_Radboudumc_Bone', 'novel_data/ULS23_Radboudumc_Pancreas', 'processed_data/fully_annotated/kits21', 'processed_data/fully_annotated/LIDC-IDRI', 'processed_data/fully_annotated/LiTS', 'processed_data/fully_annotated/MDSC_Task06_Lung', 'processed_data/fully_annotated/MDSC_Task07_Pancreas', 'processed_data/fully_annotated/MDSC_Task10_Colon', 'processed_data/fully_annotated/NIH_LN_ABD', 'processed_data/fully_annotated/NIH_LN_MED']:
        zip_files = [os.path.join(extracted_base_labels, subdir, 'labels', el) for el in os.listdir(os.path.join(extracted_base_labels, subdir, 'labels')) if '.zip' in el]
        for zip_file in zip_files:
            unzip_file(zip_file, os.path.join(extracted_base_labels, subdir, 'labels'), verbose=False)
    '''


    image_paths = []
    for subdir in ['novel_data/ULS23_DeepLesion3D', 'novel_data/ULS23_Radboudumc_Bone', 'novel_data/ULS23_Radboudumc_Pancreas', 'processed_data/fully_annotated/kits21', 'processed_data/fully_annotated/LIDC-IDRI', 'processed_data/fully_annotated/LiTS', 'processed_data/fully_annotated/MDSC_Task06_Lung', 'processed_data/fully_annotated/MDSC_Task07_Pancreas', 'processed_data/fully_annotated/MDSC_Task10_Colon', 'processed_data/fully_annotated/NIH_LN_ABD', 'processed_data/fully_annotated/NIH_LN_MED']:
        image_paths += [os.path.join(extracted_base, subdir, 'images', el) for el in os.listdir(os.path.join(extracted_base, subdir, 'images')) if '.nii.gz' in el]
    image_paths = [el for el in image_paths if os.path.exists(el.replace('/ULS23/', '/ULS23-main/annotations/ULS23/').replace('/images/', '/labels/'))]
    


    tasks = [(i, path) for i, path in enumerate(image_paths)]

    # Use partial to pass static args
    process_fn = partial(process_file_uls, extracted_base=extracted_base,
                         converted_path=converted_path, label_classes=label_classes)

    filename_map = {}


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_fn, task) for task in tasks]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing ULS images"):
            results.append(future.result())

    for filename, mapping in results:
        filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

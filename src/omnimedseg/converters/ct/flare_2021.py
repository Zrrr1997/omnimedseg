import os
import shutil
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

def process_file_flare_2021(idx_file_pair, extracted_base, converted_path, label_classes):
    idx, file = idx_file_pair
    filename = f"{idx:06d}.nii.gz"
    out_image_path = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, out_image_path)

    base_path = os.path.basename(file)
    label_file = file.replace(base_path, f'TrainingMask/{base_path}').replace('_0000', '')
    label = sitk.ReadImage(label_file)

    for label_id in np.arange(1, 5):
        label_image = sitk.Equal(label, float(label_id), 0, 1)
        label_image = sitk.Cast(label_image, sitk.sitkUInt8)
        out_file = os.path.join(converted_path, 'labels', label_classes[label_id - 1], filename)
        sitk.WriteImage(label_image, out_file)

    return filename, {
        'image': file,
        **{k: label_file for k in label_classes}
    }

def convert_flare_2021():
    converted_path = os.path.join('datasets', 'CT', 'FLARE_2021', 'converted')
    label_classes = ['liver', 'kidneys', 'spleen', 'pancreas']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'FLARE_2021', 'extracted')
    image_paths = sorted([
        os.path.join(extracted_base, f)
        for f in os.listdir(extracted_base)
        if '.nii.gz' in f
    ])

    tasks = [(i, path) for i, path in enumerate(image_paths)]

    # Use partial to pass static args
    process_fn = partial(process_file_flare_2021, extracted_base=extracted_base,
                         converted_path=converted_path, label_classes=label_classes)

    filename_map = {}

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_fn, tasks),
            total=len(tasks),
            desc="🖼️  Processing FLARE_2021 images"
        ))

    for filename, mapping in results:
        filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

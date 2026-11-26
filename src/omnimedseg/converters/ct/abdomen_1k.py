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

def process_file_abdomen_1k(idx_file_pair, extracted_base, converted_path, label_classes):
    idx, file = idx_file_pair
    filename = f"{idx:06d}.nii.gz"
    out_image_path = os.path.join(converted_path, 'images', filename)
    if os.path.exists(file):
        shutil.move(file, out_image_path)

    base_path = os.path.basename(os.path.dirname(file))
    label_file = file.replace(base_path, 'Mask').replace('_0000.nii.gz', '.nii.gz')

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

def convert_abdomen_1k():
    print(
        "\n" + "=" * 80 +
        "\n⚠️  WARNING: --keep_raw is disabled for ABDOMEN_1K (too large)\n"
        "   To reproduce the raw data, run again with --extract\n"
        "   (make sure you have enough disk space!)\n" +
        "=" * 80 + "\n"
    )
    converted_path = os.path.join(data_root(), 'CT', 'ABDOMEN_1K', 'converted')
    label_classes = ['liver', 'kidneys', 'spleen', 'pancreas']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'ABDOMEN_1K', 'extracted')
    image_paths = []
    for i in range(1, 4):
        image_paths += sorted([
            os.path.join(extracted_base, f'AbdomenCT-1K-ImagePart{i}',f)
            for f in os.listdir(os.path.join(extracted_base,  f'AbdomenCT-1K-ImagePart{i}'))
            if '.nii.gz' in f and os.path.exists(os.path.join(extracted_base, 'Mask', f.replace('_0000.nii.gz', '.nii.gz')))
        ])

    tasks = [(i, path) for i, path in enumerate(image_paths)]

    # Use partial to pass static args
    process_fn = partial(process_file_abdomen_1k, extracted_base=extracted_base,
                         converted_path=converted_path, label_classes=label_classes)

    filename_map = {}


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_fn, task) for task in tasks]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing ABDOMEN_1K images"):
            results.append(future.result())

    for filename, mapping in results:
        filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

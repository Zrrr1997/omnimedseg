from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import subprocess
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

def process_file_covid19_lung_and_infection(idx_file, converted_path, label_classes):
    idx, file = idx_file
    try:
        filename = f"{idx:06d}.nii.gz"

        # Read and write image
        image = sitk.ReadImage(file)
        sitk.WriteImage(image, os.path.join(converted_path, 'images', filename))

        # Read label
        label = sitk.ReadImage(file.replace('COVID-19-CT-Seg_20cases', 'Lung_and_Infection_Mask'))

        # Process label classes
        for label_id in np.arange(1, 4):
            label_image = sitk.Equal(label, float(label_id), 0, 1)
            label_image = sitk.Cast(label_image, sitk.sitkUInt8)
            out_file = os.path.join(converted_path, 'labels', label_classes[label_id - 1], filename)
            sitk.WriteImage(label_image, out_file)

        # Combine labels for lungs
        label_1 = sitk.Equal(label, float(1), 0, 1)
        label_2 = sitk.Equal(label, float(2), 0, 1)
        lungs = sitk.Or(label_1, label_2)
        lungs = sitk.Cast(lungs, sitk.sitkUInt8)
        sitk.WriteImage(lungs, os.path.join(converted_path, 'labels', 'lungs', filename))

        return filename, {
            'image': file,
            **{k: file.replace('COVID-19-CT-Seg_20cases', 'Lung_and_Infection_Mask') for k in label_classes}
        }
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None


def convert_covid19_lung_and_infection():
    converted_path = os.path.join(data_root(), 'CT', 'COVID19_LUNG_AND_INFECTION', 'converted')
    label_classes = ['left_lung', 'right_lung', 'lung_lesion', 'lungs']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'COVID19_LUNG_AND_INFECTION', 'extracted', 'COVID-19-CT-Seg_20cases')
    image_paths = sorted([os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if f.endswith('.nii.gz')])

    # Pair each file with a sequential index
    indexed_paths = list(enumerate(image_paths))

    filename_map = {}

    # Wrap the worker to inject shared args
    worker = partial(process_file_covid19_lung_and_infection, converted_path=converted_path, label_classes=label_classes)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(worker, indexed_paths), total=len(indexed_paths), desc="Processing COVID19_LUNG_AND_INFECTION images"))

    for result in results:
        if result:
            filename, mapping = result
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
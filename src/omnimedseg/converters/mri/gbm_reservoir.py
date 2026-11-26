from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import SimpleITK as sitk

def filter_processed(image_path):
    label_file = image_path.replace('flair', 'seg_processed').replace('t1', 'seg_processed').replace('t2', 'seg_processed').replace('t1ce', 'seg_processed')
    basename = os.path.basename(label_file)
    basename_ = basename.replace('BraTS', 'Brats')
    label_file = label_file.replace(basename, basename_)  
    return os.path.exists(label_file)

def convert_gbm_reservoir():
    converted_path = os.path.join(data_root(), 'MRI', 'GBM_RESERVOIR', 'converted')

    # Create label mapping
    label_classes = ['edema', 'enhancing_tumor', 'necrosis']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'GBM_RESERVOIR', 'extracted')
    image_paths = []
    image_paths += [os.path.join(extracted_base, f, f'{f}_flair.nii.gz') for f in os.listdir(extracted_base) if 'BraTS2021_' in f]
    image_paths += [os.path.join(extracted_base, f, f'{f}_t1.nii.gz') for f in os.listdir(extracted_base) if 'BraTS2021_' in f]
    image_paths += [os.path.join(extracted_base, f, f'{f}_t1ce.nii.gz') for f in os.listdir(extracted_base) if 'BraTS2021_' in f]
    image_paths += [os.path.join(extracted_base, f, f'{f}_t2.nii.gz') for f in os.listdir(extracted_base) if 'BraTS2021_' in f]

    image_paths = [el for el in image_paths if filter_processed(el)]

    filename_map = {}
    print("note: --keep-raw has no effect for GBM_RESERVOIR (raw data is too large to keep); "
          "run `omnimedseg extract GBM_RESERVOIR` to recreate it")

    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        if not os.path.exists(file):
            file = os.path.join(os.path.dirname(file), os.path.basename(file).replace('BraTS2021', 'Brats2021'))

        dest_image_path = os.path.join(converted_path, 'images', filename)
        
        shutil.move(file, dest_image_path)
        label_file = file.replace('flair', 'seg_processed').replace('t1', 'seg_processed').replace('t2', 'seg_processed').replace('t1ce', 'seg_processed')
        basename = os.path.basename(label_file)
        basename_ = basename.replace('BraTS', 'Brats')
        label_file = label_file.replace(basename, basename_)
        label = sitk.ReadImage(label_file)
        edema = sitk.Equal(label, float(1), 0, 1)
        enhancing_tumor = sitk.Equal(label, float(2), 0, 1)
        necrosis = sitk.Equal(label, float(3), 0, 1)

        edema = sitk.Cast(edema, sitk.sitkUInt8)
        enhancing_tumor = sitk.Cast(enhancing_tumor, sitk.sitkUInt8)
        necrosis = sitk.Cast(necrosis, sitk.sitkUInt8)
        sitk.WriteImage(edema, os.path.join(converted_path, 'labels', 'edema', filename))
        sitk.WriteImage(enhancing_tumor, os.path.join(converted_path, 'labels', 'enhancing_tumor', filename))
        sitk.WriteImage(necrosis, os.path.join(converted_path, 'labels', 'necrosis', filename))

        return filename, {'image': file, 'necrosis': label_file, 'edema': label_file, 'enhancing_tumor': label_file}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing GBM_RESERVOIR images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

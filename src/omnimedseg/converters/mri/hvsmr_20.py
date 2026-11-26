from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import SimpleITK as sitk
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_hvsmr_20():
    converted_path = os.path.join(data_root(), 'MRI', 'HVSMR_20', 'converted')

    # Create label mapping
    label_classes = ['left_ventricle', 'right_ventricle', 'left_atrium', 'right_atrium', 'aorta', 'pulmonary_atrery', 'superior_vena_cava', 'inferior_vena_cava']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'HVSMR_20', 'extracted', 'orig')

    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if 'orig.nii.gz' in f]

    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)

        label_file = file.replace('orig.nii.gz', 'orig_seg.nii.gz')
        label = sitk.ReadImage(label_file)
        label_file_endpoints = file.replace('orig.nii.gz', 'orig_seg_endpoints.nii.gz')
        label_endpoints = sitk.ReadImage(label_file_endpoints)

        arr_label = sitk.GetArrayFromImage(label)
        arr_endpoints = sitk.GetArrayFromImage(label_endpoints)

        # Union while preserving values (take max at each voxel)
        arr_union = np.maximum(arr_label, arr_endpoints)

        # Convert back to image (keep geometry from one of them)
        union_img = sitk.GetImageFromArray(arr_union.astype(np.uint8))
        union_img.CopyInformation(label)  # copy spacing, origin, direction  
        temp_dict = {'image': file}
        for i, class_label in enumerate(label_classes):
            target = sitk.Equal(union_img, float(i+1), 0, 1)  
            target = sitk.Cast(target, sitk.sitkUInt8)
            sitk.WriteImage(target, os.path.join(converted_path, 'labels', class_label, filename))
            temp_dict[class_label] = [label_file, label_file_endpoints]

        return filename, temp_dict

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing HVSMR_20 images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

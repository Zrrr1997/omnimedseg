
from omnimedseg.core.paths import data_root
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
import SimpleITK as sitk

def _process_topcow(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"
    label_classes = ['topcow_ba', 'topcow_r_pca', 'topcow_l_pca', 'topcow_r_ica', 'topcow_r_mca', 'topcow_l_ica', 'topcow_l_mca', 'topcow_r_pcom', 'topcow_l_pcom', 'topcow_acom', 'topcow_r_aca', 'topcow_l_aca', 'topcow_3rd_a2']

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)

    # Copy label
    label_file = file.replace('imagesTr', 'cow_seg_labelsTr').replace('_0000.nii.gz', '.nii.gz')
    label_img = sitk.ReadImage(label_file)
    for i, label in enumerate(label_classes):
        dst_label = os.path.join(converted_path, 'labels', label, filename)
        label_i = sitk.BinaryThreshold(label_img, lowerThreshold=i+1, upperThreshold=i+1, insideValue=1, outsideValue=0)
        sitk.WriteImage(label_i, dst_label)
    data = {'image': file}
    for i, label in enumerate(label_classes):
        data[label] = label_file

    return filename, data


def convert_topcow(num_workers=8):
    converted_path = os.path.join(data_root(), 'CT', 'TOPCOW', 'converted')
    label_classes = ['topcow_ba', 'topcow_r_pca', 'topcow_l_pca', 'topcow_r_ica', 'topcow_r_mca', 'topcow_l_ica', 'topcow_l_mca', 'topcow_r_pcom', 'topcow_l_pcom', 'topcow_acom', 'topcow_r_aca', 'topcow_l_aca', 'topcow_3rd_a2']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'TOPCOW', 'extracted', 'TopCoW2024_Data_Release', 'imagesTr')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(os.path.join(extracted_base))]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_topcow, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TOPCOW images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

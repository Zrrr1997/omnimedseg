import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils.dataset_utils import *
import SimpleITK as sitk

def _process_pengwin(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"
    label_classes = ['sacrum', 'left_hipbone', 'right_hipbone']

    dst_img = os.path.join(converted_path, 'images', filename)
    img = sitk.ReadImage(file)
    sitk.WriteImage(img, dst_img)

    label_file = file.replace('PENGWIN_CT_train_images_part1', 'PENGWIN_CT_train_labels')
    label_file = label_file.replace('PENGWIN_CT_train_images_part2', 'PENGWIN_CT_train_labels')
    label_img = sitk.ReadImage(label_file)

    for i, label in enumerate(label_classes):
        dst_label = os.path.join(converted_path, 'labels', label, filename)
        label_i = sitk.BinaryThreshold(label_img, lowerThreshold=10*i + 1, upperThreshold=10*(i+1), insideValue=1, outsideValue=0)
        sitk.WriteImage(label_i, dst_label)
    data = {'image': file}
    for i, label in enumerate(label_classes):
        data[label] = label_file

    return filename, data


def convert_pengwin(num_workers=8):
    converted_path = os.path.join('datasets', 'CT', 'PENGWIN', 'converted')
    label_classes = ['sacrum', 'left_hipbone', 'right_hipbone']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'PENGWIN', 'extracted')
    image_paths = []
    image_paths += [os.path.join(extracted_base, 'PENGWIN_CT_train_images_part1', f) for f in os.listdir(os.path.join(extracted_base, 'PENGWIN_CT_train_images_part1'))]
    image_paths += [os.path.join(extracted_base, 'PENGWIN_CT_train_images_part2', f) for f in os.listdir(os.path.join(extracted_base, 'PENGWIN_CT_train_images_part2'))]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_pengwin, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing PENGWIN images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

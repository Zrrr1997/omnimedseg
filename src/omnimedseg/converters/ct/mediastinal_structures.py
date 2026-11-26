from omnimedseg.core.paths import data_root
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
import SimpleITK as sitk

def _process_mediastinal_structures(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"
    label_classes = ['esophagus', 'azygos_vein', 'subclavian_arteries', 'brachiocephalic_veins', 'lymph_node']
    label_map = {
        'esophagus': 'Esophagus',
        'azygos_vein': 'Azygos',
        'subclavian_arteries': 'SubCarArt',
        'brachiocephalic_veins': 'BrachiocephalicVeins',
        'lymph_node': 'LymphNodes'
    }

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)
    data = {'image': file}

    for label_class in label_classes:
        label_file = file.replace('data.nii.gz', f'labels_{label_map[label_class]}.nii.gz')
        dst_label = os.path.join(converted_path, 'labels', label_class, filename)

        # For lymph_node: make binary
        if label_class == 'lymph_node':
            label_img = sitk.ReadImage(label_file)
            label_arr = sitk.GetArrayFromImage(label_img)
            label_arr[label_arr > 0] = 1  # binary threshold
            binary_label_img = sitk.GetImageFromArray(label_arr)
            binary_label_img.CopyInformation(label_img)  # preserve spacing/origin
            sitk.WriteImage(binary_label_img, dst_label)
        else:
            shutil.copy(label_file, dst_label)

        data[label_class] = dst_label

    return filename, data


def convert_mediastinal_structures():
    converted_path = os.path.join(data_root(), 'CT', 'MEDIASTINAL_STRUCTURES', 'converted')
    label_classes = ['esophagus', 'azygos_vein', 'subclavian_arteries', 'brachiocephalic_veins', 'lymph_node']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'MEDIASTINAL_STRUCTURES', 'extracted')
    image_paths = []
    image_paths += [os.path.join(extracted_base, 'Benchmark', f, f'{f.lower()}_data.nii.gz') for f in os.listdir(os.path.join(extracted_base, 'Benchmark')) if '.csv' not in f]

    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_mediastinal_structures, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing MEDIASTINAL_STRUCTURES images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

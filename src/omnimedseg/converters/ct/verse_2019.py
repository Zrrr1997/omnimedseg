from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_verse_2019():
    converted_path = os.path.join(data_root(), 'CT', 'VERSE_2019', 'converted')

    # Create label mapping
    label_map = {i: f'C{i}' if i < 8 else f'T{i}' if i < 20 else f'L{i}' if i < 26 else 'T13'
                 for i in range(1, 29) if i != 27}
    label_classes = list(label_map.values())
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'VERSE_2019', 'extracted')
    image_paths = []
    for split in ['training', 'validation', 'test']:
        split_dir = os.path.join(extracted_base, f'dataset-verse19{split}', 'rawdata')
        image_paths += glob.glob(os.path.join(split_dir, '*', '*_ct.nii.gz'))

    filename_map = {}
    split_map = defaultdict(list)

    os.makedirs(os.path.join(converted_path, 'images'), exist_ok=True)
    for cls in label_classes:
        os.makedirs(os.path.join(converted_path, 'labels', cls), exist_ok=True)

    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"
        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)

        label_file = file.replace('rawdata', 'derivatives').replace('_ct.nii.gz', '_seg-vert_msk.nii.gz')
        label = sitk.ReadImage(label_file)
        temp_dict = {'image': file}

        for label_id in np.unique(sitk.GetArrayFromImage(label)):
            if label_id == 0 or label_id not in label_map:
                continue
            label_image = sitk.Equal(label, float(label_id), 0, 1)
            label_image = sitk.Cast(label_image, sitk.sitkUInt8)

            out_file = os.path.join(converted_path, 'labels', label_map[label_id], filename)
            sitk.WriteImage(label_image, out_file)
            temp_dict[label_map[label_id]] = label_file

        if 'training' in file:
            split = 'train'
        elif 'validation' in file:
            split = 'val'
        else:
            split = 'test'

        return filename, temp_dict, split

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing VERSE_2019 images"):
            filename, temp_dict, split = f.result()
            filename_map[filename] = temp_dict
            split_map[split].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, dict(split_map), json_fn='split_map')
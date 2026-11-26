import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_picai():
    converted_path = os.path.join('datasets', 'MRI', 'PICAI', 'converted')

    # Create label mapping
    label_classes = ['prostate_lesion']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'PICAI', 'extracted')

    image_paths = []
    for i in range(10000, 11476):
        image_paths += [os.path.join(extracted_base, f'{i}', el) for el in os.listdir(os.path.join(extracted_base, f'{i}')) if 't2w.mha' in el]


    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        


        label_basepath = os.path.basename(file).replace('_t2w.mha', '.nii.gz')
        label_file = os.path.join(extracted_base, 'picai_labels-main', 'csPCa_lesion_delineations', 'human_expert', 'resampled', label_basepath)
        if os.path.exists(label_file):
            if np.sum(nib.load(label_file).get_fdata()) > 0:
                shutil.copy(label_file, os.path.join(converted_path, 'labels', 'prostate_lesion', filename))
                nii_img = sitk.ReadImage(file)
                sitk.WriteImage(nii_img, dest_image_path)

                return filename, {'image': file, 'prostate_lesion': label_file}
        return None, None

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing PICAI images"):
            filename, temp_dict = f.result()
            if filename is not None:
                filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

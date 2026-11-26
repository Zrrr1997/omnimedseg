from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_spider():
    converted_path = os.path.join(data_root(), 'MRI', 'SPIDER', 'converted')

    # Create label mapping
    label_classes = ['vertebrae', 'IVD', 'spinal_canal']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'SPIDER', 'extracted', 'images')

    image_paths = [os.path.join(extracted_base, el) for el in os.listdir(os.path.join(extracted_base)) if '.mha' in el]


    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        image = sitk.ReadImage(file)
        sitk.WriteImage(image, dest_image_path)

        label_file = file.replace('images', 'masks')
        label_img = sitk.ReadImage(label_file)
        vertebrae = sitk.BinaryThreshold(label_img, lowerThreshold=1, upperThreshold=10, insideValue=1, outsideValue=0)
        ivd = sitk.BinaryThreshold(label_img, lowerThreshold=201, upperThreshold=210, insideValue=1, outsideValue=0)
        canal = sitk.BinaryThreshold(label_img, lowerThreshold=100, upperThreshold=100, insideValue=1, outsideValue=0)
        sitk.WriteImage(vertebrae, os.path.join(converted_path, 'labels', 'vertebrae', filename))
        sitk.WriteImage(ivd, os.path.join(converted_path, 'labels', 'IVD', filename))
        sitk.WriteImage(canal, os.path.join(converted_path, 'labels', 'spinal_canal', filename))
        return filename, {'image': file, 'vertebrae': label_file, 'IVD': label_file, 'spinal_canal': label_file}

    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing SPIDER images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict


    save_mapping(converted_path, filename_map)

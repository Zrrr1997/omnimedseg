from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_ocutox():
    # Define paths
    converted_path = os.path.join(data_root(), 'Fundus', 'OCUTOX', 'converted')
    ensure_directories(converted_path, ['lesions'])

    extracted_base = os.path.join(data_root(), 'Fundus', 'OCUTOX', 'extracted')

    idx = 0
    filename_map = {}
    image_files = os.listdir(os.path.join(extracted_base, 'images'))
    for basefile in tqdm(os.listdir(os.path.join(extracted_base, 'masks')), desc="Processing OcuTox images", unit="file"):

        file = os.path.join(extracted_base, 'masks', basefile)
        image_file = [el for el in image_files if basefile.split('.')[0].replace('-a', '').replace('-i', '').replace('-2', '').replace('-3','') == el.split('.')[0]]
        if len(image_file) == 0:
            print(f"warning: could not read image for {file}")
            continue
        else:
            image_file = image_file[0]
        image = cv2.imread(os.path.join(extracted_base, 'images', image_file))
        label = cv2.imread(file)

        if image is None:
            print(f"warning: could not read image for {file}")
            continue

        filename = f"{idx:06d}.png"
        filename_map[filename] = {'image': os.path.join(extracted_base, 'images', image_file), 'lesions': file}

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'lesions', filename), label)

        idx += 1

    save_mapping(converted_path, filename_map)
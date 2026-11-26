
from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_skin_hair():
    converted_path = os.path.join(data_root(), 'Dermoscopy', 'SKIN_HAIR', 'converted')
    ensure_directories(converted_path, ['skin_hair'])

    extracted_base = os.path.join(data_root(), 'Dermoscopy', 'SKIN_HAIR', 'extracted', 'A skin lesion hair mask dataset with fine-grained annotations', 'dermoscopic_image')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]
    idx = 0

    filename_map = {}

    for file in tqdm(image_paths, desc="🖼️  Processing SKIN_HAIR images"):
        filename = f"{idx:06d}.png"

        image = cv2.imread(file)
        label_file = file.replace('dermoscopic_image', 'hair_mask')
        label = cv2.imread(label_file)

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'skin_hair', filename), label)


        filename_map[filename] = {
            'image': file,
            'skin_hair': label_file
        }
        idx += 1
    save_mapping(converted_path, filename_map)
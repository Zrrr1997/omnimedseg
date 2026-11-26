import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *


def convert_hvdropdb():
    converted_path = os.path.join('datasets', 'Fundus', 'HVDROPDB', 'converted')
    ensure_directories(converted_path, ['optic_disc', 'vessels', 'ridge'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join('datasets', 'Fundus', 'HVDROPDB', 'extracted', 'HVDROPDB_RetCam_Neo_Segmentation')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(os.path.join(extracted_base))]
    target_mapping = {'Vessels': 'vessels', 'OpticDisc': 'optic_disc', 'Ridge': 'ridge'}

    for path in tqdm(sorted(image_paths), desc="🖼️  Processing HVDROPDB images"):

        image_subpaths = [os.path.join(path, el) for el in os.listdir(path) if 'images' in el]
        for image_path in image_subpaths:
            image_files = [os.path.join(image_path, el) for el in os.listdir(image_path)]
            for image_file in image_files:
                filename = f"{idx:06d}.png"
                img = cv2.imread(image_file)
                label = cv2.imread(image_file.replace('_images', '_masks'))
                target = target_mapping[image_path.split('_')[-2]]

                cv2.imwrite(os.path.join(converted_path, 'images', filename), img)
                cv2.imwrite(os.path.join(converted_path, 'labels', target, filename), label)
                filename_map[filename] = {'image': image_file, target: image_file.replace('_images', '_masks')}
                idx += 1
    save_mapping(converted_path, filename_map)
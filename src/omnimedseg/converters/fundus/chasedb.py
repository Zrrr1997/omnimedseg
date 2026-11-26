from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_chasedb():
    converted_path = os.path.join(data_root(), 'Fundus', 'CHASEDB', 'converted')
    ensure_directories(converted_path, ['vessels'])
    idx = 0
    filename_map = {}

    extracted_base = os.path.join(data_root(), 'Fundus', 'CHASEDB', 'extracted')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if 'HO' not in f]
    os.makedirs(os.path.join(converted_path, 'labels', 'vessels_annotator_1'), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'labels', 'vessels_annotator_2'), exist_ok=True)

    for file in tqdm(image_files, desc="Processing CHASEDB images"):
        filename = f"{idx:06d}.png"

        image = cv2.imread(file)
        vessels_exp1 = cv2.imread(file.replace('.jpg', '_1stHO.png'))
        vessels_exp2 = cv2.imread(file.replace('.jpg', '_2ndHO.png'))
        label = (((vessels_exp1 > 0) + (vessels_exp2 > 0)) > 0).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels_annotator_1', filename), vessels_exp1)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels_annotator_2', filename), vessels_exp2)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels', filename), label)
        filename_map[filename] = {
            'image': file,
            'vessels_annotator_1': file.replace('.jpg', '_1stHO.png'),
            'vessels_annotator_2': file.replace('.jpg', '_2ndHO.png'),
            'vessels': [file.replace('.jpg', '_1stHO.png'), file.replace('.jpg', '_2ndHO.png')]
        }
        idx += 1
    save_mapping(converted_path, filename_map)
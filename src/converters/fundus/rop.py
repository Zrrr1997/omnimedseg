import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *

def convert_rop():
    converted_path = os.path.join('datasets', 'Fundus', 'ROP', 'converted')
    ensure_directories(converted_path, ['optic_disc'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join('datasets', 'Fundus', 'ROP', 'extracted')
    image_files = [os.path.join(extracted_base, 'Optic disc segmentation(ROP)', f) for f in os.listdir(os.path.join(extracted_base, 'Optic disc segmentation(ROP)')) if '.jpg' in f]

    for file in tqdm(sorted(image_files), desc="🖼️  Processing ROP images"):
        filename = f"{idx:06d}.png"
        shutil.copy(file, os.path.join(converted_path, 'images', filename))
        label_path = file.replace('.jpg', '.json')
        with open(label_path, 'r') as f:
            label_json = json.load(f)
        contour = np.array(label_json['shapes'][0]['points'])
        shape = (label_json['imageHeight'], label_json['imageWidth'], 3)
        optic_disc = (contour_to_mask(contour, shape) > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_disc', filename), optic_disc)

        filename_map[filename] = {'image': file, 'optic_disc': file.replace('.jpg', '.json')}
        idx += 1
    save_mapping(converted_path, filename_map)
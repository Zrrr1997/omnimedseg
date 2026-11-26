import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(idx_file):
    idx, file, converted_path = idx_file
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('/originalImages/', '/semanticMasks/').replace('.jpg', '_mask.tif')
    label = cv2.imread(label_file)
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'artefact', filename), label)
    return filename, file, label_file

def convert_ead():
    converted_path = os.path.join('datasets', 'Endoscopy', 'EAD', 'converted')
    ensure_directories(converted_path, ['artefact'])
    extracted_base = os.path.join('datasets', 'Endoscopy', 'EAD', 'extracted', 'EAD2020_train', 'EAD2020-Phase-II-Detection_Segmentation-99Frames', 'originalImages')
    image_files = [os.path.join(extracted_base,  f) for f in os.listdir(extracted_base)]
    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, (idx, file, converted_path)) for idx, file in enumerate(image_files)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing EAD images"):
            filename, file, label_file = f.result()
            filename_map[filename] = {
                'image': file,
                'artefact': label_file,
            }
    save_mapping(converted_path, filename_map)

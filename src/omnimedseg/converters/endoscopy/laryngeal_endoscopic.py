from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_image(idx_file):
    idx, file, converted_path, label_classes = idx_file
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)
    label_file = file.replace('/img/', '/annot/')
    label = cv2.imread(label_file)

    vocal_folds = (label == 1).astype(np.uint8) * 255
    tissue = (label == 2).astype(np.uint8) * 255
    glottal_space = (label == 3).astype(np.uint8) * 255
    tumor = (label == 4).astype(np.uint8) * 255
    instrument = (label == 5).astype(np.uint8) * 255
    intubation = (label == 6).astype(np.uint8) * 255
    label_mapping = {
        'vocal_folds': vocal_folds,
        'tissue': tissue,
        'glottal_space': glottal_space,
        'tumor': tumor,
        'instrument': instrument,
        'intubation': intubation
    }

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    for label_class in label_classes:
        cv2.imwrite(os.path.join(converted_path, 'labels', label_class, filename), label_mapping[label_class])

    return filename, file, label_classes, label_file

def convert_laryngeal_endoscopic():
    converted_path = os.path.join(data_root(), 'Endoscopy', 'LARYNGEAL_ENDOSCOPIC', 'converted')
    label_classes = ['vocal_folds', 'tissue', 'glottal_space', 'tumor', 'instrument', 'intubation']
    ensure_directories(converted_path, label_classes)
    extracted_base = os.path.join(data_root(), 'Endoscopy', 'LARYNGEAL_ENDOSCOPIC', 'extracted', 'vocalfolds-master', 'img')
    image_files = []

    for i in np.arange(1, 5):
        image_files += [os.path.join(extracted_base, 'patient1', f'seq{i}', f) for f in os.listdir(os.path.join(extracted_base, 'patient1', f'seq{i}'))]
    for i in np.arange(5, 9):
        image_files += [os.path.join(extracted_base, 'patient2', f'seq{i}', f) for f in os.listdir(os.path.join(extracted_base, 'patient2', f'seq{i}'))]

    filename_map = {}
    args = [(idx, file, converted_path, label_classes) for idx, file in enumerate(image_files)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, arg) for arg in args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing LARYNGEAL_ENDOSCOPIC images"):
            filename, file, label_classes, label_file = f.result()
            filename_map[filename] = {
                'image': file,
                **{k: label_file for k in label_classes}
            }

    save_mapping(converted_path, filename_map)
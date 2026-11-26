from omnimedseg.core.paths import data_root
import os
import cv2
import xmltodict
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *

def process_file(file, extracted_base, converted_path):
    try:
        filename = f"{os.path.basename(file).split('.')[0]}.png"
        image = cv2.imread(file)
        shape = image.shape

        label_file = os.path.join(extracted_base, f'{os.path.basename(file).split("_")[0]}.xml')
        with open(label_file, 'r', encoding='utf-8') as f:
            xml_dict = xmltodict.parse(f.read())

            if type(xml_dict['case']['mark']) != list:
                if xml_dict['case']['mark']['svg'] is None:
                    return None
                contour = np.array([[el['x'], el['y']] for el in eval(xml_dict['case']['mark']['svg'])[0]['points']])
                label = (contour_to_mask(contour, shape) > 0).astype(np.uint8) * 255
            else:
                index = int(os.path.basename(file).split('_')[-1].split('.')[0])
                gt = xml_dict['case']['mark'][index-1]
                if gt['svg'] is None:
                    return None
                contour = np.array([[el['x'], el['y']] for el in eval(gt['svg'])[0]['points']])
                label = (contour_to_mask(contour, shape) > 0).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'thyroid_tumor', filename), label)

        return filename, file, label_file
    except Exception as e:
        return None

def convert_ddti():
    converted_path = os.path.join(data_root(), 'Ultrasound', 'DDTI', 'converted')
    ensure_directories(converted_path, ['thyroid_tumor'])
    extracted_base = os.path.join(data_root(), 'Ultrasound', 'DDTI', 'extracted')

    image_files = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if '.xml' not in f]
    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, extracted_base, converted_path) for file in image_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DDTI images"):
            result = future.result()
            if result:
                filename, file, label_file = result
                filename_map[filename] = {
                    'image': file,
                    'thyroid_tumor': label_file
                }

    save_mapping(converted_path, filename_map)

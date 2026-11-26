import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from omnimedseg.utils.dataset_utils import *

def process_image(args):
    idx, file, class_names, converted_path = args
    filename = f"{idx:06d}.png"
    with suppress_stderr():
        image = cv2.imread(file)
        label_file = file.replace('EMDS5-Original', 'EMDS5-Ground Truth').replace('.png', '-GTM.png')
        label = cv2.imread(label_file)
    class_index = int(label_file.split('/')[-2])
    class_name = class_names[class_index]
    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'microorganism', filename), label)
    return filename, class_name, file, label_file

def convert_emds_6():
    converted_path = os.path.join('datasets', 'Microscopy', 'EMDS_6', 'converted')
    ensure_directories(converted_path, ['microorganism'])
    class_names = {
        1: "actinophrys",
        2: "arcella",
        3: "aspidisca",
        4: "codosiga",
        5: "colpoda",
        6: "epistylis",
        7: "euglypha",
        8: "paramecium",
        9: "rotifera",
        10: "vorticella",
        11: "noctiluca",
        12: "ceratium",
        13: "stentor",
        14: "siprostomum",
        15: "keratella quadrala",
        16: "euglena",
        17: "gymnodinium",
        18: "gonyaulax",
        19: "phacus",
        20: "stylongchia",
        21: "synchaeta"
    }
    class_map = {name: [] for name in class_names.values()}

    extracted_base = os.path.join('datasets', 'Microscopy', 'EMDS_6', 'extracted', 'EMDS-6', 'EMDS5-Original')
    image_paths = []
    for index in range(1, 22):
        image_paths += [os.path.join(extracted_base, f"{index:02d}", f) for f in os.listdir(os.path.join(extracted_base, f"{index:02d}"))]
    
    filename_map = {}

    args_list = [(idx, file, class_names, converted_path) for idx, file in enumerate(image_paths)]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, args_list), total=len(args_list), desc="🖼️  Processing EMDS_6 images"))

    for filename, class_name, file, label_file in results:
        class_map[class_name].append(filename)
        filename_map[filename] = {
            'image': file,
            'microorganism': label_file
        }

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, class_map, json_fn='class_map')
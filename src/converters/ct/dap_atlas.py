import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils.dataset_utils import *
from src.converters.utils_ct import *
import pandas as pd

def remap_labels(label_classes):
    new_label_classes = []
    for el in label_classes:
        el = el.replace(' ', '_')
        el = el.replace('bowel', 'intestine')
        if '_left' in el:
            el = el.replace('_left', '')
            el = 'left_' + el
        if '_right' in el:
            el = el.replace('_right', '')
            el = 'right_' + el
        el = el.replace('uterocervix', 'utero_cervix')
        el = el.replace('vertebrae_', '')
        if 'costa' in el:
            el = 'ribs'
        el = el.replace('_hip', '_hipbone')
        if 'portal' in el:
            el = 'portal_vein_splenic_vein'
        el = el.replace('___', '')
        new_label_classes.append(el)
    return new_label_classes
        

def convert_dap_atlas():
    converted_path = os.path.join('datasets', 'CT', 'DAP_ATLAS', 'converted')

    label_map = pd.read_csv('utils/conversion/dap_atlas_map.csv')


    label_classes = label_map[' Name'].values[2:]
    label_indices = label_map['Label'].values[2:]
    label_classes = remap_labels(label_classes)

    index_to_class = dict(zip(label_indices, label_classes))

    ensure_directories(converted_path, np.unique(label_classes))

    extracted_base = os.path.join('datasets', 'CT', 'DAP_ATLAS', 'extracted', 'Atlas_final_dataset_V1_533')
    if not os.path.exists(os.path.join('datasets/CT/AUTOPET_2024_CT/extracted')):
        raise FileNotFoundError("Please download and extract the AUTOPET_2024_CT dataset first, as DAP_ATLAS depends on it. \n You can do this by running: \n python main_script.py --dataset AUTOPET_2024_CT --download --extract")

    autopet_base = os.path.join('datasets/CT/AUTOPET_2024_CT/extracted/imagesTr')

    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if '.nii.gz' in f]
    image_paths_filtered = []
    for image_path in image_paths:
        identifier = os.path.basename(image_path).split('_')[-1].split('.')[0]
        img_file = [os.path.join(autopet_base, el) for el in os.listdir(autopet_base) if identifier in el and '_0000' in el]
        if len(img_file) == 1:
            image_paths_filtered.append(image_path)
    image_paths = image_paths_filtered


    args_list = [(file, idx, converted_path, index_to_class) for idx, file in enumerate(image_paths)]


    # Replace this block in convert_dap_atlas():
    filename_map = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for filename, data in tqdm(executor.map(process_dap_atlas_case, args_list),
                                total=len(args_list),
                                desc="🖼️ Processing DAP_ATLAS images"):
            filename_map[filename] = data


    save_mapping(converted_path, filename_map)

def process_dap_atlas_case(args):
    file, idx, converted_path, index_to_class = args

    filename = f"{idx:06d}.nii.gz"
    label_img = sitk.ReadImage(file)
    
    identifier = os.path.basename(file).split('_')[-1].split('.')[0]
    autopet_base = os.path.join('datasets/CT/AUTOPET_2024_CT/extracted/imagesTr')
    img_file = [os.path.join(autopet_base, el) for el in os.listdir(autopet_base) if identifier in el and '_0000' in el][0]

    shutil.move(img_file, os.path.join(converted_path, 'images', filename))
    

    for label in index_to_class:
        if label == 67:
            label_i = sitk.BinaryThreshold(label_img, lowerThreshold=67, upperThreshold=90, insideValue=1, outsideValue=0)
            sitk.WriteImage(label_i, os.path.join(converted_path, 'labels', 'ribs', filename))
        elif label > 67 and label <= 90:
            continue
        else:
            label_i = sitk.BinaryThreshold(label_img, lowerThreshold=int(label), upperThreshold=int(label), insideValue=1, outsideValue=0)
            sitk.WriteImage(label_i, os.path.join(converted_path, 'labels', index_to_class[label], filename))


    return filename, {
        'image': 'img_file',
        **{cls: file 
           for cls in np.unique(list(index_to_class.values()))}
    }

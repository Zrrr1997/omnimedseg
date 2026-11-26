from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *
import pandas as pd
def convert_covid19_cxr():
    converted_path = os.path.join(data_root(), 'XRay', 'COVID19_CXR', 'converted')
    labels = [
    "left_lung",
    "right_lung",
    "lungs",
    "cardiomediastinum",
    "airways",
    "ground_glass_opacities",
    "consolidation",
    "pleural_effusion",
    "pneumothorax",
    "endotracheal_tube",
    "central_venous_line",
    "monitoring_probes",
    "nosogastric_tube",
    "tubings"
]
    ensure_directories(converted_path, labels)

    extracted_base = os.path.join(data_root(), 'XRay', 'COVID19_CXR', 'extracted', 'covid-19-chest-xray-segmentations-dataset-master')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    idx = 0
    filename_map = {}

    df = pd.read_csv(os.path.join(extracted_base, "structure.csv"))
    df = df.groupby("Image").agg({
        "Mask": list,
        "Template Name": list,
        "Filename": list,
        'Top': list,
        'Left': list,
        'Width': list,
        'Height': list
    }).reset_index()

    df_meta = pd.read_csv(os.path.join(extracted_base, "metadata.csv"))
    image_fns = df_meta['filename'].tolist()
    licenses = df_meta['license'].tolist()
    image_license_map = dict(zip(image_fns, licenses))
    # Iterate over rows
    for index, row in tqdm(df.iterrows(), desc="Processing COVID19_CXR images"):
        image_path = os.path.join(extracted_base, 'images', row["Filename"][0])
        if image_license_map.get(os.path.basename(image_path), "unknown") in ["nan", "CC-BY-NC-ND"]:
            print(f"warning: skipping {image_path} due to license restrictions.")
            continue
        filename = f"{idx:06d}.png"
        image = cv2.imread(image_path)
        class_names = row["Template Name"]
        masks = row["Mask"]
        temp_dict = {'image': image_path}
        lungs = np.zeros(image.shape[:2]).astype(np.uint8)
        for i in range(len(class_names)):
            if type(masks[i]) != str:
                continue
            mask_path = os.path.join(extracted_base, masks[i])

            label = np.zeros(image.shape[:2]).astype(np.uint8)
            label_file = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) * 255
            label[row['Top'][i]:row['Top'][i]+row['Height'][i], row['Left'][i]:row['Left'][i]+row['Width'][i]] = label_file
            label_file = label
            class_name = class_names[i].lower().replace(' ', '_')
            if 'lung' in class_name:
                lungs += label_file
                if 'lungs' in temp_dict.keys():
                    temp_dict['lungs'].append(mask_path)
                else:
                    temp_dict['lungs'] = [mask_path]

            cv2.imwrite(os.path.join(converted_path, 'labels', class_name, filename), label_file)
            temp_dict[class_name] = mask_path
        if 'lungs' in temp_dict.keys():
            cv2.imwrite(os.path.join(converted_path, 'labels', 'lungs', filename), lungs)


        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)

        filename_map[filename] = temp_dict
        idx += 1
    save_mapping(converted_path, filename_map)
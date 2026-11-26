import os
import cv2
import numpy as np
from omnimedseg.utils.dataset_utils import *

def process_file_ham_10000_nevus(args):
    idx, file, converted_path, class_mapping, class_name = args
    filename = f"{idx:06d}.png"
    try:
        image = cv2.imread(file)
        label_file = file.replace('.jpg', '.jpg.json').replace('img', 'ann')
        label = (np.array(json_mask_to_png(label_file, class_mapping)) > 0).astype(np.uint8) * 255


        img_path = os.path.join(converted_path, 'images', filename)
        lbl_path = os.path.join(converted_path, 'labels', class_name, filename)
        cv2.imwrite(img_path, image)
        cv2.imwrite(lbl_path, label)

        return filename, file, label_file
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None
        
def load_class_mapping(classes_json_path):
    with open(classes_json_path, "r") as f:
        config = json.load(f)
    return {
        class_def["title"]: idx + 1  # start at 1 to avoid background=0
        for idx, class_def in enumerate(config.get("classes", []))
        if class_def.get("shape") == "bitmap"
    }
from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_akram():
    # Define paths
    converted_path = os.path.join(data_root(), 'Fundus', 'AKRAM', 'converted')
    ensure_directories(converted_path, ['vessels', 'arteries', 'veins'])

    extracted_base = os.path.join(
        data_root(), 'Fundus', 'AKRAM', 'extracted',
        'Data on Fundus Images for Vessels Segmentation, Detection of Hypertensive Retinopathy, Diabetic Retinopathy and Papilledema'
    )
    extract_archives(extracted_base)

    av_data_path = os.path.join(extracted_base, 'AV')

    idx = 0
    filename_map = {}

    for folder in tqdm(os.listdir(av_data_path), desc="Processing AKRAM images"):
        img_dir = os.path.join(av_data_path, folder)
        if not os.path.isdir(img_dir):
            continue

        all_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        if folder == 'IM000189':
            all_files = [os.path.join(img_dir, 'IM000189', f) for f in os.listdir(os.path.join(img_dir, 'IM000189'))]

        vessels = find_label_file(all_files, 'vessels', ends_with='jpg')
        arteries = find_label_file(all_files, 'art', ends_with='jpg') or find_label_file(all_files, 'ater', ends_with='jpg')
        veins = find_label_file(all_files, 'vei', ends_with='jpg')

        image = vessels.replace('--vessels', '') if vessels else None
        image = find_label_file(all_files, os.path.basename(image).split('.')[0] + '.', ends_with='jpg') if image else None

        paths = {
            'image': image,
            'vessels': vessels,
            'arteries': arteries,
            'veins': veins,
        }

        filename = f"{idx:06d}.png"
        filename_map[filename] = {k: v for k, v in paths.items() if v}

        for label, path in paths.items():
            img = cv2.imread(path)

            if label in ['arteries', 'veins']:
                img_mask = ~(img == 255).all(axis=-1)
                img = img_mask.astype('uint8') * 255

            if label == 'vessels':
                img = 255 - img

            out_dir = os.path.join(converted_path, 'images' if label == 'image' else os.path.join('labels', label))
            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, img)
        idx += 1
    save_mapping(converted_path, filename_map)

# Utility Functions
def find_label_file(files, keyword, ends_with=None):
    matches = [f for f in files if keyword.lower() in os.path.basename(f).lower()]
    if ends_with:
        matches = [f for f in matches if f.lower().endswith(ends_with.lower())]
    return matches[0] if matches else None
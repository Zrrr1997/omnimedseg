from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_csc():
    # Define paths
    converted_path = os.path.join(data_root(), 'Fundus', 'CSC', 'converted')
    ensure_directories(converted_path, ['subretinal_fluid'])

    extracted_base = os.path.join(data_root(), 'Fundus', 'CSC', 'extracted', '4k64fwnp4k-5')
    extract_archives(extracted_base, pattern='All_graders_train_test')

    idx = 0
    filename_map = {}

    split_map = {'train': [], 'test': []}

    for split in ['train', 'test']:
        extracted_split = os.path.join(extracted_base, split)
        for file in tqdm(os.listdir(extracted_split), desc=f"🖼️  Processing CSC {split} images", unit="file"):
            image_file = os.path.join(extracted_split, file)
            filename = f"{idx:06d}.png"
            filename_map[filename] = {
                label: os.path.join(extracted_base, split, file) for label in ['image', 'subretinal_fluid']
            }
            split_map[split.lower()].append(filename)

            image_label = cv2.imread(image_file)
            half_width = int(image_label.shape[1] / 2)
            image = image_label[:, 0:half_width, :]
            label = image_label[:, half_width:2*half_width, :]
            tolerance = 240

            # A pixel is "white" if all its RGB values are >= tolerance
            is_almost_white = np.all(label >= tolerance, axis=-1)

            # Binary mask: 1 where not almost white, 0 where almost white
            label = (~is_almost_white).astype(np.uint8) * 255
            label[:15, :] = 0
            label[-15:, :] = 0
            label[:, :15] = 0
            label[:, -15:] = 0

            cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
            cv2.imwrite(os.path.join(converted_path, 'labels', 'subretinal_fluid', filename), label)

            idx += 1

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn="split_map")
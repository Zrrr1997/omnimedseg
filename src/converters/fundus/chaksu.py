import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_chaksu():
    converted_path = os.path.join('datasets', 'Fundus', 'CHAKSU', 'converted')
    ensure_directories(converted_path, ['optic_disc', 'optic_cup'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join('datasets', 'Fundus', 'CHAKSU', 'extracted')
    split_map = {'train': [], 'test': []}

    for i in np.arange(1, 6):
        os.makedirs(os.path.join(converted_path, 'labels', f'optic_disc_annotator_{i}'), exist_ok=True)
        os.makedirs(os.path.join(converted_path, 'labels', f'optic_cup_annotator_{i}'), exist_ok=True)

    for split in ['Train', 'Test']:
        directory = os.path.join(extracted_base, split)
        # List all matching files
        image_files = []
        for root, _, files in os.walk(directory):
            if '1.0' not in root:
                continue
            for file in files:
                if file.endswith(('JPG', 'jpg', 'png')):
                    image_files.append(os.path.join(root, file))

        def process_image(file_idx_tuple):
            file, idx = file_idx_tuple
            filename = f"{idx:06d}.png"
            result = {}

            image = cv2.imread(file)
            if image is None:
                return None  # Skip failed image

            label_path_exp_oc = []
            label_path_exp_od = []

            for i in range(1, 6):
                cup_file = file.replace('1.0_Original_Fundus_Images', f'3.0_Doctors_Annotations_Binary_OD_OC/Expert {i}')
                cup_file = cup_file.replace('Bosch/', 'Bosch/Cup/').replace('Forus/', 'Forus/Cup/').replace('Remidio/', 'Remidio/Cup/')
                cup_file = cup_file.replace('JPG', 'tif').replace('jpg', 'tif').replace('png', 'tif')
                disc_file = cup_file.replace('Cup', 'Disc')

                if not os.path.exists(cup_file):
                    cup_file = cup_file.replace('tif', 'png')
                if not os.path.exists(disc_file):
                    disc_file = disc_file.replace('tif', 'png')

                label_path_exp_oc.append(cup_file)
                label_path_exp_od.append(disc_file)

            label_od = file.replace('1.0_Original_Fundus_Images', '5.0_OD_OC_Mean_Median_Majority_STAPLE')
            label_od = label_od.replace('Bosch/', 'Bosch/Disc/Mean/').replace('Forus/', 'Forus/Disc/Mean/').replace('Remidio/', 'Remidio/Disc/Mean/')
            label_od = label_od.replace('JPG', 'tif').replace('jpg', 'tif').replace('png', 'tif')
            label_cup = label_od.replace('Disc', 'Cup')

            if not os.path.exists(label_cup):
                label_cup = label_cup.replace('tif', 'png')
            if not os.path.exists(label_od):
                label_od = label_od.replace('tif', 'png')

            # Save files
            cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
            cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_disc', filename), cv2.imread(label_od))
            cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_cup', filename), cv2.imread(label_cup))

            for i in range(1, 6):
                if os.path.exists(label_path_exp_od[i-1]):
                    continue
                cv2.imwrite(os.path.join(converted_path, 'labels', f'optic_disc_annotator_{i}', filename), cv2.imread(label_path_exp_od[i-1]))
                cv2.imwrite(os.path.join(converted_path, 'labels', f'optic_cup_annotator_{i}', filename), cv2.imread(label_path_exp_oc[i-1]))

            filename_dict = {
                'image': file,
                'optic_disc': label_od,
                'optic_cup': label_cup,
            }
            for i in range(1, 6):
                filename_dict[f'optic_disc_annotator_{i}'] = label_path_exp_od[i-1]
                filename_dict[f'optic_cup_annotator_{i}'] = label_path_exp_oc[i-1]

            result['filename'] = filename
            result['filename_dict'] = filename_dict
            return result

        # Parallel execution
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_image, (file, idx)) for idx, file in enumerate(sorted(image_files))]
            for future in tqdm(as_completed(futures), total=len(image_files), desc=f"🖼️  Processing CHAKSU {split} images"):
                res = future.result()
                if res:
                    split_map[split.lower()].append(res['filename'])
                    filename_map[res['filename']] = res['filename_dict']
    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')
import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_papila():
    converted_path = os.path.join('datasets', 'Fundus', 'PAPILA', 'converted')
    ensure_directories(converted_path, ['optic_disc', 'optic_cup'])
    idx = 0
    filename_map = {}
    extracted_base = os.path.join('datasets', 'Fundus', 'PAPILA', 'extracted')

    image_files = [os.path.join(extracted_base, 'PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f', 'FundusImages', f) for f in os.listdir(os.path.join(extracted_base, 'PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f', 'FundusImages'))]
    os.makedirs(os.path.join(converted_path, 'labels', 'optic_disc_annotator_1'), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'labels', 'optic_disc_annotator_2'), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'labels', 'optic_cup_annotator_1'), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'labels', 'optic_cup_annotator_2'), exist_ok=True)
    indexed_files = list(enumerate(image_files))

    def process_image(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.png"

        img = cv2.imread(file)
        if img is None:
            print(f"❌ Could not load {file}")
            return None

        img_shape = img.shape

        label_path = file.replace('FundusImages/', 'ExpertsSegmentations/Contours/')
        label_path_exp_1_oc = label_path.replace('.jpg', '_cup_exp1.txt')
        label_path_exp_1_od = label_path.replace('.jpg', '_disc_exp1.txt')
        label_path_exp_2_oc = label_path.replace('.jpg', '_cup_exp2.txt')
        label_path_exp_2_od = label_path.replace('.jpg', '_disc_exp2.txt')

        optic_disc_exp_1 = (contour_to_mask(label_path_exp_1_od, img_shape) > 0).astype(np.uint8) * 255
        optic_cup_exp_1 = (contour_to_mask(label_path_exp_1_oc, img_shape) > 0).astype(np.uint8) * 255
        optic_disc_exp_2 = (contour_to_mask(label_path_exp_2_od, img_shape) > 0).astype(np.uint8) * 255
        optic_cup_exp_2 = (contour_to_mask(label_path_exp_2_oc, img_shape) > 0).astype(np.uint8) * 255

        optic_disc = (((optic_disc_exp_1 > 0) + (optic_disc_exp_2 > 0)) > 0).astype(np.uint8) * 255
        optic_cup = (((optic_cup_exp_1 > 0) + (optic_cup_exp_2 > 0)) > 0).astype(np.uint8) * 255

        shutil.copy(file, os.path.join(converted_path, 'images', filename))
        cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_disc_annotator_1', filename), optic_disc_exp_1)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_disc_annotator_2', filename), optic_disc_exp_2)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_cup_annotator_1', filename), optic_cup_exp_1)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_cup_annotator_2', filename), optic_cup_exp_2)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_disc', filename), optic_disc)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_cup', filename), optic_cup)

        return filename, {
            'image': file,
            'optic_disc_annotator_1': label_path_exp_1_od,
            'optic_disc_annotator_2': label_path_exp_2_od,
            'optic_cup_annotator_1': label_path_exp_1_oc,
            'optic_cup_annotator_2': label_path_exp_2_oc,
            'optic_disc': [label_path_exp_1_od, label_path_exp_2_od],
            'optic_cup': [label_path_exp_1_oc, label_path_exp_2_oc]
        }
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_image, idx_file) for idx_file in indexed_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing PAPILA images"):
            result = future.result()
            if result is not None:
                filename, data = result
                filename_map[filename] = data
    save_mapping(converted_path, filename_map)
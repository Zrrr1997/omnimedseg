import os 
import cv2
from tqdm import tqdm
from collections import defaultdict
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_dermoscopy import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_isic_2018():
    base_dir = os.path.join('datasets', 'Dermoscopy', 'ISIC_2018')
    converted_path = os.path.join(base_dir, 'converted')
    ensure_directories(converted_path, ['lesion', 'globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks'])

    extracted_base = os.path.join(base_dir, 'extracted')
    subdirs = [d for d in os.listdir(extracted_base) if '_Input' in d]

    image_paths = []
    seen_ids = set()

    for subdir in subdirs:
        full_path = os.path.join(extracted_base, subdir)
        for fname in os.listdir(full_path):
            if fname not in seen_ids and '.jpg' in fname:
                seen_ids.add(fname)
                image_paths.append(os.path.join(full_path, fname))

    # Preload label ID lists
    label_lists = {
        'test_task': set(os.listdir(os.path.join(extracted_base, 'ISIC2018_Task1-2_Test_Input'))),

        'train_task': set(os.listdir(os.path.join(extracted_base, 'ISIC2018_Task1-2_Training_Input'))),

        'val_task': set(os.listdir(os.path.join(extracted_base, 'ISIC2018_Task1-2_Validation_Input'))),

    }

    filename_map = {}
    split_map = defaultdict(list)

    def process_image(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.png"
        image_id = os.path.basename(file)
        temp_dict = {'image': file}

        image = cv2.imread(file)
        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)

        def save_label(label_type, source_path):
            label_img = cv2.imread(source_path)
            if label_img is not None and np.sum(label_img) != 0:
                out_path = os.path.join(converted_path, 'labels', label_type, filename)
                cv2.imwrite(out_path, label_img)
                temp_dict[label_type] = source_path
        if image_id in label_lists['train_task']:
            lesion_path = os.path.join(extracted_base, 'ISIC2018_Task1_Training_GroundTruth',
                                       image_id.replace('.jpg', '_segmentation.png'))
            save_label('lesion', lesion_path)
            for label in ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']:
                label_file = os.path.join(extracted_base, 'ISIC2018_Task2_Training_GroundTruth_v3',
                                           image_id.replace('.jpg', f'_attribute_{label}.png'))
                save_label(label, label_file)
            split_map['train'].append(filename)

        if image_id in label_lists['test_task']:
            lesion_path = os.path.join(extracted_base, 'ISIC2018_Task1_Test_GroundTruth',
                                       image_id.replace('.jpg', '_segmentation.png'))
            save_label('lesion', lesion_path)
            for label in ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']:
                label_file = os.path.join(extracted_base, 'ISIC2018_Task2_Test_GroundTruth',
                                           image_id.replace('.jpg', f'_attribute_{label}.png'))
                save_label(label, label_file)
            split_map['test'].append(filename)
        if image_id in label_lists['val_task']:
            lesion_path = os.path.join(extracted_base, 'ISIC2018_Task1_Validation_GroundTruth',
                                       image_id.replace('.jpg', '_segmentation.png'))
            save_label('lesion', lesion_path)
            for label in ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']:
                label_file = os.path.join(extracted_base, 'ISIC2018_Task2_Validation_GroundTruth',
                                           image_id.replace('.jpg', f'_attribute_{label}.png'))
                save_label(label, label_file)
            split_map['val'].append(filename)



        return filename, temp_dict

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, (i, path)): i for i, path in enumerate(image_paths)}
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="🖼️  Processing ISIC_2018 images"):
            filename, temp_dict = future.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)
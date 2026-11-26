import os 
import cv2
from tqdm import tqdm
from collections import defaultdict
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_dermoscopy import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_isic_2017():
    base_dir = os.path.join('datasets', 'Dermoscopy', 'ISIC_2017')
    converted_path = os.path.join(base_dir, 'converted')
    ensure_directories(converted_path, ['lesion'])

    extracted_base = os.path.join(base_dir, 'extracted')
    subdirs = [d for d in os.listdir(extracted_base) if '_Data' in d]

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
        'test': set(os.listdir(os.path.join(extracted_base, 'ISIC-2017_Test_v2_Data'))),
        'train': set(os.listdir(os.path.join(extracted_base, 'ISIC-2017_Training_Data'))),
        'val': set(os.listdir(os.path.join(extracted_base, 'ISIC-2017_Validation_Data'))),
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

        if image_id in label_lists['train']:
            lesion_path = os.path.join(extracted_base, 'ISIC-2017_Training_Part1_GroundTruth',
                                       image_id.replace('.jpg', '_segmentation.png'))
            save_label('lesion', lesion_path)
            split_map['train'].append(filename)

        elif image_id in label_lists['test']:
            lesion_path = os.path.join(extracted_base, 'ISIC-2017_Test_v2_Part1_GroundTruth',
                                       image_id.replace('.jpg', '_segmentation.png'))
            save_label('lesion', lesion_path)
            split_map['test'].append(filename)

        elif image_id in label_lists['val']:
            lesion_path = os.path.join(extracted_base, 'ISIC-2017_Validation_Part1_GroundTruth',
                                       image_id.replace('.jpg', '_segmentation.png'))
            save_label('lesion', lesion_path)
            split_map['val'].append(filename)
        else:
            print(f'Error image {image_id} not found in any directory!')


        return filename, temp_dict

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, (i, path)): i for i, path in enumerate(image_paths)}
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="🖼️ Processing ISIC_2017 images"):
            filename, temp_dict = future.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)
from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from collections import defaultdict
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_dermoscopy import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_isic_2016():
    base_dir = os.path.join(data_root(), 'Dermoscopy', 'ISIC_2016')
    converted_path = os.path.join(base_dir, 'converted')
    ensure_directories(converted_path, ['lesion', 'globules', 'streaks'])

    extracted_base = os.path.join(base_dir, 'extracted')
    subdirs = [d for d in os.listdir(extracted_base) if '_Data' in d]

    image_paths = []
    seen_ids = set()

    for subdir in subdirs:
        full_path = os.path.join(extracted_base, subdir)
        for fname in os.listdir(full_path):
            if fname not in seen_ids and 'Segmentation' not in fname and 'superpixels' not in fname:
                seen_ids.add(fname)
                image_paths.append(os.path.join(full_path, fname))

    # Preload label ID lists
    label_lists = {
        'part1_test': set(os.listdir(os.path.join(extracted_base, 'ISBI2016_ISIC_Part1_Test_Data'))),
        'part1_train': set(os.listdir(os.path.join(extracted_base, 'ISBI2016_ISIC_Part1_Training_Data'))),
        'part2b_test': set(os.listdir(os.path.join(extracted_base, 'ISBI2016_ISIC_Part2B_Test_Data'))),
        'part2b_train': set(os.listdir(os.path.join(extracted_base, 'ISBI2016_ISIC_Part2B_Training_Data')))
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

        if image_id in label_lists['part1_train']:
            lesion_path = os.path.join(extracted_base, 'ISBI2016_ISIC_Part1_Training_GroundTruth',
                                       image_id.replace('.jpg', '_Segmentation.png'))
            save_label('lesion', lesion_path)
            split_map['train'].append(filename)

        elif image_id in label_lists['part1_test']:
            lesion_path = os.path.join(extracted_base, 'ISBI2016_ISIC_Part1_Test_GroundTruth',
                                       image_id.replace('.jpg', '_Segmentation.png'))
            save_label('lesion', lesion_path)
            split_map['test'].append(filename)

        if image_id in label_lists['part2b_train']:
            prefix = os.path.join(extracted_base, 'ISBI2016_ISIC_Part2B_Training_GroundTruth',
                                  image_id.replace('.jpg', ''))
            save_label('globules', f"{prefix}_globules.png")
            save_label('streaks', f"{prefix}_streaks.png")
            split_map['train'].append(filename)

        elif image_id in label_lists['part2b_test']:
            prefix = os.path.join(extracted_base, 'ISBI2016_ISIC_Part2B_Test_GroundTruth',
                                  image_id.replace('.jpg', ''))
            save_label('globules', f"{prefix}_globules.png")
            save_label('streaks', f"{prefix}_streaks.png")
            split_map['train'].append(filename)

        return filename, temp_dict

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, (i, path)): i for i, path in enumerate(image_paths)}
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="🖼️ Processing ISIC_2016 images"):
            filename, temp_dict = future.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)
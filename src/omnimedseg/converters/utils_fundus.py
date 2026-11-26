import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from omnimedseg.utils.dataset_utils import *

def convert_train_test_dataset(extracted_base, converted_path, split_data, convention_map, label_name_extension=''):
    filename_map = {}
    split_map = {'train': [], 'test': []}

    all_args = []
    idx = 0

    for split in split_data:
        extracted_split = os.path.join(extracted_base, split, convention_map['image'])
        files = [f for f in os.listdir(extracted_split) if f.endswith(('png', 'jpg'))]
        for file in files:
            all_args.append((idx, file, split, extracted_base, converted_path, convention_map, label_name_extension))
            idx += 1

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, all_args), total=len(all_args), desc="Processing images"))

    for result in results:
        filename = result['filename']
        filename_map[filename] = result['filename_map_entry']
        split_map[result['split']].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn="split_map")


def process_file(args):
    idx, file, split, extracted_base, converted_path, convention_map, label_name_extension = args

    image_dir = os.path.join(extracted_base, split, convention_map['image'])
    label_dir = os.path.join(extracted_base, split, convention_map['vessels'])

    image_file = os.path.join(image_dir, file)

    file_stem, _ = os.path.splitext(file)
    label_file_name = file_stem + label_name_extension + '.png'
    label_file = os.path.join(label_dir, label_file_name)
    if '/ORVS/' in label_file:
        label_file = label_file.replace('.png', '.jpg')

    filename = f"{idx:06d}.png"

    result = {}

    if image_file.endswith(('.png', '.jpg')):
        img = cv2.imread(image_file)
        if img is not None:
            cv2.imwrite(os.path.join(converted_path, 'images', filename), img)
        label = cv2.imread(label_file)
        if label is not None:
            cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels', filename), label)
        else:
            print(f"warning: could not read label: {label_file}")

    result['filename'] = filename
    result['split'] = split.lower()
    result['filename_map_entry'] = {
        label: os.path.join(extracted_base, split, convention_map[label], file.replace('.jpg', f'{label_name_extension}.jpg').replace('.png', f'{label_name_extension}.png')) for label in ['vessels']
    }
    result['filename_map_entry'] = {
        label: os.path.join(extracted_base, split, convention_map[label], file) for label in ['image']
    }

    return result




if __name__ == "__main__":
    pass

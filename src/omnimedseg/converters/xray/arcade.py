from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def process_image_task(args):
    idx, file, class_label, split, class_map = args
    filename = f"{idx:06d}.png"
    image = cv2.imread(file)

    label_file = os.path.join(data_root(), 'XRay', 'ARCADE', 'extracted', 'arcade', f'{class_label}', f'{split}', 'annotations', f'{split}.json')
    with open(label_file, encoding="utf-8") as f:
        gt = json.load(f)

    im_anns_gt = defaultdict(list)
    for ann in gt["annotations"]:
        im_anns_gt[ann["image_id"]].append(ann)

    gt_mask = np.zeros((512, 512), np.int32)
    for id, im in im_anns_gt.items():
        if str(id) != os.path.basename(file).split('.')[0]:
            continue
        for ann in im:
            points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], np.int32).T
            points = points.reshape((-1, 1, 2))
            tmp = np.zeros((512, 512), np.int32)
            cv2.fillPoly(tmp, [points], (1))
            gt_mask += tmp
            

    label = (gt_mask > 0).astype(np.uint8) * 255

    return {
        'filename': filename,
        'image': image,
        'label': label,
        'class_key': class_map[class_label],
        'split': split,
        'file': file,
        'label_file': label_file
    }


def convert_arcade():
    base_path = os.path.join(data_root(), 'XRay', 'ARCADE')
    converted_path = os.path.join(base_path, 'converted')
    ensure_directories(converted_path, ['coronary_artery', 'stenosis'])

    os.makedirs(os.path.join(converted_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'labels', 'coronary_artery'), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'labels', 'stenosis'), exist_ok=True)

    idx = 0
    class_map = {'stenosis': 'stenosis', 'syntax': 'coronary_artery'}
    tasks = []

    for class_label in ['syntax', 'stenosis']:
        for split in ['train', 'val', 'test']:
            extracted_base = os.path.join(base_path, 'extracted', 'arcade', f'{class_label}', f'{split}', 'images')
            image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]
            for file in image_paths:
                tasks.append((idx, file, class_label, split, class_map))
                idx += 1

    results = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_image_task, tasks), total=len(tasks), desc="🖼️  Processing ARCADE images"):
            results.append(result)

    filename_map = {}
    split_map = {'train': [], 'val': [], 'test': []}
    for result in results:
        filename = result['filename']
        cv2.imwrite(os.path.join(converted_path, 'images', filename), result['image'])
        cv2.imwrite(os.path.join(converted_path, 'labels', result['class_key'], filename), result['label'])

        filename_map[filename] = {
            'image': result['file'],
            result['class_key']: result['label_file']
        }
        split_map[result['split']].append(filename)

    save_mapping(converted_path, filename_map)
    save_mapping(converted_path, split_map, json_fn='split_map')
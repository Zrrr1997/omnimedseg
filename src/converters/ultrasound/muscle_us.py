import os
import cv2
import numpy as np
from tqdm import tqdm
from src.utils.dataset_utils import *
from concurrent.futures import ProcessPoolExecutor

def process_image_muscle_us(args):
    idx, file, muscle, converted_path, muscle_map = args
    filename = f"{idx:06d}.png"

    image = cv2.imread(file)
    label = cv2.imread(file.replace('/Images/', '/Masks/'))

    image_path = os.path.join(converted_path, 'images', filename)
    label_path = os.path.join(converted_path, 'labels', muscle_map[muscle], filename)

    cv2.imwrite(image_path, image)
    cv2.imwrite(label_path, label)

    return filename, file, muscle_map[muscle]


def convert_muscle_us():

    converted_path = os.path.join('datasets', 'Ultrasound', 'MUSCLE_US', 'converted')
    ensure_directories(converted_path, ['biceps_brachii', 'tibialis_anterior', 'gastrocnemius_medialis'])
    extracted_base = os.path.join('datasets', 'Ultrasound', 'MUSCLE_US', 'extracted', 'Polito-Radboud-DeepLearningUS')

    muscle_map = {'BB': 'biceps_brachii', 'GM': 'gastrocnemius_medialis', 'TA': 'tibialis_anterior'}
    filename_map = {}

    tasks = []
    idx = 0

    for muscle in ['BB', 'GM', 'TA']:
        base_path = os.path.join(extracted_base, muscle)
        image_files_healthy = [os.path.join(base_path, 'Healthy', 'Images', f) for f in os.listdir(os.path.join(base_path, 'Healthy', 'Images'))]
        image_files_pathological = [os.path.join(base_path, 'Pathological', 'Images', f) for f in os.listdir(os.path.join(base_path, 'Pathological', 'Images'))]
        image_files = image_files_healthy + image_files_pathological

        for file in image_files:
            tasks.append((idx, file, muscle, converted_path, muscle_map))
            idx += 1

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_image_muscle_us, tasks), total=len(tasks), desc="🖼️  Processing MuscleUS images..."):
            filename, file, muscle_dir = result
            filename_map[filename] = {
                'image': file,
                muscle_dir: file.replace('Images', 'Masks')
            }

    save_mapping(converted_path, filename_map)
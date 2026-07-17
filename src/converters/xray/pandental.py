import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.dataset_utils import *

def segment_object_robust(image_path, patch_size=5, color_thresh=30, morph_kernel=5, show=False):
    img = cv2.imread(image_path)

    h, w = img.shape[:2]

    # Get average background color from a small top-center patch
    y0 = 0
    x0 = w // 2
    half = patch_size // 2
    bg_patch = img[max(0, y0-half):y0+patch_size, max(0, x0-half):x0+patch_size]
    bg_color = np.mean(bg_patch.reshape(-1, 3), axis=0)

    # Compute Euclidean distance from background color
    diff = np.linalg.norm(img.astype(np.float32) - bg_color.reshape(1, 1, 3), axis=2)

    # Threshold difference
    mask = (diff > color_thresh).astype(np.uint8) * 255

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=5)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(mask, contours, i, 255, -1)
    # Apply mask to image
    object_only = (cv2.bitwise_and(img, img, mask=mask) > 0).astype(np.uint8) * 255

    return object_only



def convert_pandental():
    converted_path = os.path.join('datasets', 'XRay', 'PANDENTAL', 'converted')
    ensure_directories(converted_path, ['mandible', 'mandible_annotator_1', 'mandible_annotator_2'])

    extracted_base = os.path.join('datasets', 'XRay', 'PANDENTAL', 'extracted', 'Images')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    filename_map = {}

    def process_file(idx, file):
        filename = f"{idx:06d}.png"
        image = cv2.imread(file)
        if '115' not in file:
            label_annotator_1 = segment_object_robust(file.replace('Images', 'Segmentation1'))
            label_annotator_2 = segment_object_robust(file.replace('Images', 'Segmentation2'))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(label_annotator_1, cv2.MORPH_OPEN, kernel)
            label_annotator_1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            label_annotator_1 = (cv2.imread(file.replace('Images', 'Segmentation1')) > 0).astype(np.uint8)
            label_annotator_2 = (cv2.imread(file.replace('Images', 'Segmentation2')) > 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            label_annotator_1 = cv2.morphologyEx(label_annotator_1, cv2.MORPH_CLOSE, kernel).astype(np.uint8) * 255
            label_annotator_2 = cv2.morphologyEx(label_annotator_2, cv2.MORPH_CLOSE, kernel).astype(np.uint8) * 255

        label = (((label_annotator_1 > 0) + (label_annotator_2 > 0)) > 0).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'mandible', filename), label)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'mandible_annotator_1', filename), label_annotator_1)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'mandible_annotator_2', filename), label_annotator_2)

        return filename, {
            'image': file,
            'mandible': [file.replace('Images', 'Segmentation1'), file.replace('Images', 'Segmentation2')],
            'mandible_annotator_1': file.replace('Images', 'Segmentation1'),
            'mandible_annotator_2': file.replace('Images', 'Segmentation2')
        }

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, idx, file): idx for idx, file in enumerate(image_paths)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing PANDENTAL images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)
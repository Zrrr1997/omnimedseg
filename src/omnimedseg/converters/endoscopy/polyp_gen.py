import os 
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from omnimedseg.utils.dataset_utils import *

def process_file_polyp_gen(args):
    converted_path = os.path.join('datasets', 'Endoscopy', 'POLYP_GEN', 'converted')
    idx, file, label_file, is_negative = args
    filename = f"{idx:06d}.png"

    image = cv2.imread(file)
    if is_negative:
        label = np.zeros_like(image).astype(np.uint8)
    else:
        label = (cv2.imread(label_file) > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'polyp', filename), label)

    return filename, {
        'image': file,
        'polyp': label_file if not is_negative else file
    }

def convert_polyp_gen():
    extracted_base = os.path.join('datasets', 'Endoscopy', 'POLYP_GEN', 'extracted', 'PolypGen2021_MultiCenterData_v3')
    tasks = []
    idx = 0

    converted_path = os.path.join('datasets', 'Endoscopy', 'POLYP_GEN', 'converted')
    ensure_directories(converted_path, ['polyp'])



    # Center images
    for c in np.arange(1, 7):
        center = os.path.join(extracted_base, f'data_C{c}', f'images_C{c}')
        image_files = [os.path.join(center, f) for f in os.listdir(center)]
        for file in image_files:
            label_file = file.replace('/images_', '/masks_').replace(']', '').replace('.jpg', '_mask.jpg')
            tasks.append((idx, file, label_file, False))
            idx += 1

    # Positive sequences
    for s in np.arange(1, 24):
        sequence = os.path.join(extracted_base, 'sequenceData', 'positive', f'seq{s}', f'images_seq{s}')
        image_files = [os.path.join(sequence, f) for f in os.listdir(sequence)]
        for file in image_files:
            label_file = file.replace('/images_seq', '/masks_seq').replace('.jpg', '_mask.jpg')
            tasks.append((idx, file, label_file, False))
            idx += 1

    # Negative sequences
    for s in np.arange(1, 24):
        sequence = os.path.join(extracted_base, 'sequenceData', 'negativeOnly', f'seq{s}_neg')
        image_files = [os.path.join(sequence, f) for f in os.listdir(sequence)]
        for file in image_files:
            tasks.append((idx, file, None, True))
            idx += 1

    filename_map = {}
    with Pool(processes=os.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file_polyp_gen, tasks), total=len(tasks), desc="Processing POLYP_GEN"):
            filename, mapping = result
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *

def convert_fundus_av_seg():
    # Define paths
    converted_path = os.path.join(data_root(), 'Fundus', 'FUNDUS_AV_SEG', 'converted')
    ensure_directories(converted_path, ['arteries', 'veins', 'vessels'])
    idx = 0

    filename_map = {}

    extracted_base = os.path.join(data_root(), 'Fundus', 'FUNDUS_AV_SEG', 'extracted', 'Fundus-AVSeg')
    images_dir = os.path.join(extracted_base, 'images')
    annotations_dir = os.path.join(extracted_base, 'annotation')

    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]

    # Define how extracted labels are mapped
    labels = ['image', 'vessels', 'arteries', 'veins']
    folder_names_extracted = {
        'image': 'images',
        'vessels': 'annotation',
        'arteries': 'annotation',
        'veins': 'annotation',
    }
    paths_extracted = {
        label: os.path.join(extracted_base, folder_names_extracted[label]) for label in labels
    }

    for image_file in tqdm(image_files, desc="🖼️  Processing FUNDUS AV SEG images"):
        img_filename = os.path.basename(image_file)
        filename = f"{idx:06d}.png"

        # Copy image to converted/images
        shutil.copy(image_file, os.path.join(converted_path, 'images', filename))

        # Read annotation
        label_path = os.path.join(annotations_dir, img_filename)
        label = cv2.imread(label_path)

        if label is None:
            print(f"⚠️ Could not read label for {img_filename}")
            continue

        veins = blue_part(label) | green_part(label)
        arteries = red_part(label) | green_part(label)
        vessels = ((label > 0).any(axis=2)).astype(np.uint8) * 255

        filename_map[filename] = {
            label: os.path.join(paths_extracted[label], os.path.basename(image_file)) for label in labels
        }

        cv2.imwrite(os.path.join(converted_path, 'labels', 'veins', filename), veins)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'arteries', filename), arteries)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'vessels', filename), vessels)
        idx += 1
    save_mapping(converted_path, filename_map)


def blue_part(label):
    return ((label[:, :, 0] > 0) & (label[:, :, 1] == 0) & (label[:, :, 2] == 0)).astype(np.uint8) * 255
def green_part(label):
    return ((label[:, :, 1] > 0) & (label[:, :, 0] == 0) & (label[:, :, 2] == 0)).astype(np.uint8) * 255
def red_part(label):
    return ((label[:, :, 2] > 0) & (label[:, :, 0] == 0) & (label[:, :, 1] == 0)).astype(np.uint8) * 255
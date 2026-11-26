import base64
import io
import json
import os
import shutil
import sys
import cv2
import zlib
from contextlib import contextmanager

import numpy as np
import skimage.draw as drw
from PIL import Image

from omnimedseg.utils.postprocess_download import is_compressed_file, is_unzipped, unzip_file

def check_filename_map(name, modality):
    base_path = os.path.join('datasets', modality, name, 'converted')
    filename_map_path = os.path.join(base_path, 'filename_map.json')

    if not os.path.exists(filename_map_path):
        print(f'❌ Filename map file {filename_map_path} does not exist!')
        return

    with open(filename_map_path, 'r') as f:
        filename_map = json.load(f)

    missing = 0

    for filename, paths in filename_map.items():
        image_path = os.path.join(base_path, 'images', filename)

        if not os.path.exists(image_path):
            print(f'❌ Missing image file: {image_path}')
            missing += 1

        for label_type, label_path in paths.items():
            if type(label_path) == str:
                if not os.path.exists(label_path):
                    print(f'❌ Missing label "{label_type}" file: {label_path}')
                    missing += 1
            else:
                for label in label_path:
                    if not os.path.exists(label):
                        print(f'❌ Missing label "{label_type}" file: {label}')
                        missing += 1
    if missing == 0:
        print(f"✅ All files in filename_map.json for {name} are accounted for.")
    else:
        print(f"⚠️ Total missing files: {missing} for {name}. Please check if the conversion of {name} was complete!")
        exit()

def remove_raw_data(name, modality):
    base_path = os.path.join('datasets', modality, name, 'extracted')
    if os.path.exists(base_path):
        print(f"🗑️  Removing raw data at {base_path}")
        shutil.rmtree(base_path)
    else:
        print(f"⚠️  Raw data path does not exist: {base_path}. Nothing to do...")

def filter_empty_labels(name, modality):
    base_path = os.path.join('datasets', modality, name, 'converted')
    filename_map_path = os.path.join(base_path, 'filename_map.json')

    if not os.path.exists(filename_map_path):
        print(f'❌ Filename map file {filename_map_path} does not exist!')
        return

    with open(filename_map_path, 'r') as f:
        filename_map = json.load(f)
    new_filename_map = {}

    filtered_count = 0

    labels = os.listdir(os.path.join(base_path, 'labels'))

    for filename, paths in filename_map.items():
        image_path = os.path.join(base_path, 'images', filename)
        present_labels = {label: True for label in labels}
        for label in labels:
            label_path = os.path.join(base_path, 'labels', label, filename)
            label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label_image is not None and np.sum(label_image) > 0:
                continue
            else:
                present_labels[label] = False
                os.remove(label_path)
                filtered_count += 1
        if all(not present for present in present_labels.values()):
            os.remove(image_path)
        else:
            new_filename_map[filename] = {
                'image': image_path,
                **{label: image_path.replace('images', f'labels/{label}') for label in labels if present_labels.get(label, False)}
            }
    save_mapping(base_path, new_filename_map)


    print(f"✅ Filtered out {filtered_count} samples with empty labels from {name}.")


def extract_archives(data_path, pattern=None):
    zip_files = [f for f in os.listdir(data_path) if is_compressed_file(os.path.join(data_path, f))]
    if pattern is not None:
        zip_files = [f for f in zip_files if pattern in f]
    print("🧾 Found compressed files:", zip_files)
    for zip_file in zip_files:
        zip_path = os.path.join(data_path, zip_file)
        if is_unzipped(data_path, zip_path):
            print(f"✅ Already extracted: {zip_file}")
        else:
            print(f"📦 Extracting: {zip_file}")
            unzip_file(zip_path, extract_to=data_path)

def ensure_directories(base_path, classes):
    paths = [
        base_path,
        os.path.join(base_path, 'images'),
        os.path.join(base_path, 'labels')
    ]
    for cls in classes:
        paths.append(os.path.join(base_path, 'labels', cls))

    for path in paths:
        os.makedirs(path, exist_ok=True)

def save_mapping(converted_path, filename_map, json_fn="filename_map"):
    # Apply sorting logic
    if json_fn == "filename_map":
        # Sort the entire dict by keys
        filename_map = dict(sorted(filename_map.items()))
    elif 'split' in json_fn:
        # Sort the values (assumed to be lists) for each key
        for key in filename_map:
            if isinstance(filename_map[key], list):
                filename_map[key] = sorted(filename_map[key])
    map_path = os.path.join(converted_path, f'{json_fn}.json')
    with open(map_path, 'w') as f:
        json.dump(filename_map, f, indent=2)
    print(f"\n📄 Saved filename map to: {map_path}")

def contour_to_mask(cont, img_shape):
    """Return mask given a contour and the shape of image
    """
    c = np.loadtxt(cont) if type(cont) == str else cont
    mask = np.zeros(img_shape[:-1], dtype=np.uint8)
    rr, cc = drw.polygon(c[:,1], c[:,0])
    rr = np.clip(rr, 0, mask.shape[0] - 1)  
    cc = np.clip(cc, 0, mask.shape[1] - 1) 
    mask[rr, cc] = 1
    return mask

@contextmanager
def suppress_stderr():
    stderr_fileno = sys.stderr.fileno()
    sys.stderr.flush()
    old_stderr = os.dup(stderr_fileno)
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), stderr_fileno)
    try:
        yield
    finally:
        os.dup2(old_stderr, stderr_fileno)
        os.close(old_stderr)

def json_mask_to_png(json_filename, class_mapping=None):
    with open(json_filename, "r") as f:
        data = json.load(f)

    width = data["size"]["width"]
    height = data["size"]["height"]
    mask = Image.new("L", (width, height), 0)

    for obj in data.get("objects", []):
        bitmap = obj.get("bitmap")
        class_title = obj.get("classTitle")

        if not bitmap or "data" not in bitmap or "origin" not in bitmap:
            continue
        if class_mapping is None or class_title not in class_mapping:
            continue

        try:
            compressed_data = base64.b64decode(bitmap["data"])
            decompressed_data = zlib.decompress(compressed_data)
            bitmap_img = Image.open(io.BytesIO(decompressed_data)).convert("L")

            # Assign pixel value according to class ID
            class_id = class_mapping[class_title]
            bitmap_arr = bitmap_img.point(lambda p: class_id if p > 0 else 0)

            origin_x, origin_y = bitmap["origin"]
            mask.paste(bitmap_arr, (origin_x, origin_y), bitmap_img)

        except Exception as e:
            print(f"❌ Error in {json_filename}: {e}")
            continue

    return mask

def load_class_mapping(classes_json_path):
    with open(classes_json_path, "r") as f:
        config = json.load(f)
    return {
        class_def["title"]: idx + 1  # start at 1 to avoid background=0
        for idx, class_def in enumerate(config.get("classes", []))
        if class_def.get("shape") == "bitmap"
    }

if __name__ == "__main__":
    pass
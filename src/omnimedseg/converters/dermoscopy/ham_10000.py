from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_dermoscopy import *
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(idx, file, converted_path):
    """Process one image + label and return mapping entry."""
    filename = f"{idx:06d}.png"

    image = cv2.imread(file)
    label_file = file.replace(
        'extracted/', 
        'extracted/HAM10000_segmentations_lesion_tschandl/'
    ).replace('.jpg', '_segmentation.png')
    label = cv2.imread(label_file)

    cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'lesion', filename), label)

    return filename, {'image': file, 'lesion': label_file}


def convert_ham_10000(workers=8):
    converted_path = os.path.join(data_root(), 'Dermoscopy', 'HAM_10000', 'converted')
    ensure_directories(converted_path, ['lesion'])

    extracted_base = os.path.join(data_root(), 'Dermoscopy', 'HAM_10000', 'extracted')
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if f.endswith('.jpg')]

    filename_map = {}

    # parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_file, idx, file, converted_path): (idx, file)
            for idx, file in enumerate(image_paths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️   Processing HAM_10000..."):
            try:
                filename, mapping = future.result()
                filename_map[filename] = mapping
            except Exception as e:
                idx, file = futures[future]
                tqdm.write(f"❌ Failed {file}: {e}")

    save_mapping(converted_path, filename_map)
    print(f"✅ Done! Processed {len(filename_map)} images.")
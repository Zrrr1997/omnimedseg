from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
from multiprocessing import Pool, cpu_count

def process_image(file, converted_path, filename_map, idx):
    filename = f"{idx:06d}.nii.gz"
    shutil.copy(file, os.path.join(converted_path, 'images', filename))
    shutil.copy(file.replace('_ct', '_seg'), os.path.join(converted_path, 'labels', 'lung_lesion', filename))

    filename_map[filename] = {
        'image': file,
        'lung_lesion': file.replace('_ct', '_seg')
    }
    return filename_map

def convert_covid19_lung_lesion():
    converted_path = os.path.join(data_root(), 'CT', 'COVID19_LUNG_LESION', 'converted')
    label_classes = ['lung_lesion']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'COVID19_LUNG_LESION', 'extracted', 'COVID-19-20_v2')
    image_paths = [os.path.join(extracted_base, 'Train', f) for f in os.listdir(os.path.join(extracted_base, 'Train')) if '_ct' in f]

    filename_map = {}

    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.starmap(
                process_image,
                [(file, converted_path, {}, idx) for idx, file in enumerate(image_paths)]
            ),
            desc="Processing COVID19_LUNG_LESION images",
            total=len(image_paths)
        ))

    for result in results:
        filename_map.update(result)

    save_mapping(converted_path, filename_map)
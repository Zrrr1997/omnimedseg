from omnimedseg.core.paths import data_root
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
import SimpleITK as sitk

def _process_curvas(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    # Copy image
    dst_img = os.path.join(converted_path, "images", filename)
    shutil.copy(file, dst_img)

    # Organ definitions (label value → organ name)
    organs = {
        1: "pancreas",
        2: "kidneys",
        3: "liver",
    }

    # Initialize union masks
    unions = {name: None for name in organs.values()}

    # Process annotations
    for i in range(1, 4):
        label_file = file.replace("image.nii.gz", f"annotation_{i}.nii.gz")
        label = sitk.ReadImage(label_file)

        for value, name in organs.items():
            mask = sitk.BinaryThreshold(
                label,
                lowerThreshold=value,
                upperThreshold=value,
                insideValue=1,
                outsideValue=0,
            )

            # Save per-annotator mask
            out_dir = os.path.join(converted_path, "labels", f"{name}_annotator_{i}")
            os.makedirs(out_dir, exist_ok=True)
            sitk.WriteImage(mask, os.path.join(out_dir, filename))

            # Update union
            unions[name] = mask if unions[name] is None else sitk.Or(unions[name], mask)

    # Save union masks
    for name, mask in unions.items():
        out_dir = os.path.join(converted_path, "labels", name)
        os.makedirs(out_dir, exist_ok=True)
        sitk.WriteImage(mask, os.path.join(out_dir, filename))

    # Build return dictionary
    result = {"image": file}
    for name in organs.values():
        for i in range(1, 4):
            result[f"{name}_annotator_{i}"] = file.replace(
                "image.nii.gz", f"annotation_{i}.nii.gz"
            )
        result[name] = [
            file.replace("image.nii.gz", f"annotation_{i}.nii.gz") for i in range(1, 4)
        ]

    return filename, result


def convert_curvas(num_workers=8):
    converted_path = os.path.join(data_root(), 'CT', 'CURVAS', 'converted')
    label_classes = ['pancreas', 'kidneys', 'liver']
    ensure_directories(converted_path, label_classes)

    for label in label_classes:
        for i in range(1, 4):
            os.makedirs(os.path.join(converted_path, 'labels', f'{label}_annotator_{i}'), exist_ok=True)

    extracted_base = os.path.join(data_root(), 'CT', 'CURVAS', 'extracted', 'training_set')
    image_paths = [os.path.join(extracted_base, f, 'image.nii.gz') for f in os.listdir(os.path.join(extracted_base))]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_curvas, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing CURVAS images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

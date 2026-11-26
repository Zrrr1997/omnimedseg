from omnimedseg.core.paths import data_root
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
import subprocess
from scipy.io import loadmat
import nibabel as nib
import numpy as np

def _process_fumpe(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"

    output_dir = file.replace('/PAT0', '/nii_PAT0')
    os.makedirs(output_dir, exist_ok=True)
    
    subprocess.run([
        "dcm2niix", "-z", "y", "-o", output_dir, "-f", "%p_%s", file
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    nii_img = [os.path.join(output_dir, el) for el in os.listdir(output_dir) if 'nii.gz' in el][0]

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(nii_img, dst_img)

    # Copy label
    label_file = file.replace('extracted/', 'extracted/label-edited/') + '.mat'
    try:
        label_img = loadmat(label_file, struct_as_record=False, squeeze_me=True)
        label_img = label_img['Mask'].astype(np.uint8)
        label_img = np.rot90(label_img, k=3, axes=(0, 1))

        ref_img = nib.load(nii_img)
        final_img = nib.Nifti1Image(label_img, affine=ref_img.affine, header=ref_img.header)
        nib.save(final_img, os.path.join(converted_path, 'labels', 'pulmonary_embolism', filename))
    except:
        print('failed to load the mat file')


    return filename, {
        'image': file,
        'pulmonary_embolism': label_file
    }


def convert_fumpe(num_workers=8):
    converted_path = os.path.join(data_root(), 'CT', 'FUMPE', 'converted')
    label_classes = ['pulmonary_embolism']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'FUMPE', 'extracted')

    image_paths = [os.path.join(extracted_base, f)
                    for f in os.listdir(os.path.join(extracted_base)) if 'label-edited' not in f and 'nii_' not in f]


    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_fumpe, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing FUMPE images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

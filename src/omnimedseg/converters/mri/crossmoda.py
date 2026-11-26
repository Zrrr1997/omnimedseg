import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

def convert_crossmoda():
    converted_path = os.path.join('datasets', 'MRI', 'CROSSMODA', 'converted')
    label_classes = ['vestibular_shwannoma', 'cochlea']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'CROSSMODA', 'extracted')
    compressed_base = os.path.join('datasets', 'MRI', 'CROSSMODA', 'compressed')
    #unzip_tcia(extracted_base, compressed_base)
    #convert_extracted_dicoms('CROSSMODA', extracted_base, converted_path)
    #convert_rtstructs('CROSSMODA', converted_path, 'CT-SEG')

    match_and_save_niftis_crossmoda()


def process_patient(patient, idx_start, converted_path, image_dest):
    """
    Process a single patient folder:
    - Match skull and tumor structures
    - Pick T1 or T2 based on shape match
    - Copy images and labels
    - Build filename mapping
    """
    filename_map = {}
    idx = idx_start

    log_file = os.path.join(patient, "log.txt")
    if not os.path.exists(log_file):
        return filename_map, idx

    with open(log_file, "r") as f:
        lines = f.readlines()

    for i in range(2):
        subdir = os.path.join(patient, f"{i}")
        if not os.path.exists(subdir):
            continue

        nii_files = [
            os.path.join(subdir, el)
            for el in os.listdir(subdir)
            if "Struct" not in el and ("t1" in el or "t2" in el)
        ]
        rt_files = [
            os.path.join(subdir, el)
            for el in os.listdir(subdir)
            if "Struct" in el
        ]
        if not rt_files:
            continue

        # Check for skull
        skulls = [el for el in rt_files if "kull" in el]
        if not skulls:
            continue
        skull = skulls[0]
        skull_shape = nib.load(skull).get_fdata().shape

        # Match T1 or T2
        t1 = next((el for el in nii_files if "t1" in el), None)
        t2 = next((el for el in nii_files if "t2" in el), None)
        nii_file, t = None, None
        if t1 and nib.load(t1).get_fdata().shape == skull_shape:
            nii_file, t = t1, "t1"
        elif t2 and nib.load(t2).get_fdata().shape == skull_shape:
            nii_file, t = t2, "t2"
        if nii_file is None:
            continue

        # Tumor
        vs = [
            el for el in rt_files
            if any(tag in el for tag in ["TV.nii.gz", "tv.nii.gz", "AN.nii.gz", "Men.nii.gz", "tumor.nii.gz"])
        ]
        if not vs:
            print(patient, "no tumor?")
            continue
        vs = vs[0]

        # Cochlea (optional)
        cochlea = next((el for el in rt_files if "ochlea.nii.gz" in el), None)

        # Copy + filename mapping
        filename = f"{idx:06d}.nii.gz"
        shutil.copy(nii_file, os.path.join(image_dest, filename))

        if t == 't1':
            shutil.copy(vs, os.path.join(converted_path, "labels", "vestibular_shwannoma", filename))
        elif cochlea and t == 't2':
            shutil.copy(cochlea, os.path.join(converted_path, "labels", "cochlea", filename))
            label_source_cochlea = [el.rstrip().split("->")[0] for el in lines if f"/{i}/" in el][0]
        if not cochlea:
            label_source_cochlea = None

        image_source = [el.rstrip().split("->")[0] for el in lines if os.path.basename(nii_file) in el][0]
        temp_dict = {"image": image_source}

        if t == 't1':
            label_source_vs = [el.rstrip().split("->")[0] for el in lines if f"/{i}/" in el][0]
            temp_dict["vestibular_shwannoma"] = label_source_vs
        elif cochlea and t == 't2':
            temp_dict["cochlea"] = label_source_cochlea

        filename_map[filename] = temp_dict
        idx += 1

    return filename_map, idx


def match_and_save_niftis_crossmoda():
    converted_path = os.path.join("datasets", "MRI", "CROSSMODA", "converted")
    all_patients = os.path.join(converted_path, "all_niftis")
    patients = [os.path.join(all_patients, el) for el in os.listdir(all_patients)]

    image_dest = os.path.join(converted_path, "images")
    filename_map = {}
    idx = 0

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_patient, patient, idx + j, converted_path, image_dest): patient
            for j, patient in enumerate(patients)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing patients"):
            try:
                patient_map, new_idx = future.result()
                filename_map.update(patient_map)
                idx = max(idx, new_idx)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    save_mapping(converted_path, filename_map)

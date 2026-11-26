import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_pancreas_ct():
    converted_path = os.path.join('datasets', 'CT', 'PANCREAS_CT', 'converted')
    label_classes = ['pancreas']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join('datasets', 'CT', 'PANCREAS_CT', 'compressed')
    extracted_base = os.path.join('datasets', 'CT', 'PANCREAS_CT', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('PANCREAS_CT', extracted_base, converted_path)
    match_pancreas_ct_img_labels()

def match_pancreas_ct_img_labels():
    idx = 0
    base_path = 'datasets/CT/PANCREAS_CT/converted/all_niftis'
    converted_path = os.path.join('datasets', 'CT', 'PANCREAS_CT', 'converted')
    filename_map = {}
    for patient in os.listdir(base_path):
        filename = f"{idx:06d}.nii.gz"
        img_file = os.path.join(base_path, patient, f'{patient}.nii.gz')
        label_file = os.path.join('datasets/CT/PANCREAS_CT/extracted/TCIA_pancreas_labels-02-05-2017', f'label{patient.split("_")[1]}.nii.gz')
        if os.path.exists(img_file) and os.path.exists(label_file):
            shutil.copy(img_file, os.path.join(converted_path, 'images', filename))
            shutil.copy(label_file, os.path.join(converted_path, 'labels', 'pancreas', filename))
            idx += 1
        filename_map[filename] = {'image': img_file, 'pancreas': label_file}
    save_mapping(converted_path, filename_map)
            



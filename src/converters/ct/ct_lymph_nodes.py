import os
from src.utils.dataset_utils import *
from src.converters.utils_ct import *
import subprocess
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import shutil

def convert_ct_lymph_nodes():
    converted_path = os.path.join('datasets', 'CT', 'CT_LYMPH_NODES', 'converted')
    label_classes = ['lymph_node']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join('datasets', 'CT', 'CT_LYMPH_NODES', 'compressed')
    extracted_base = os.path.join('datasets', 'CT', 'CT_LYMPH_NODES', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('CT_LYMPH_NODES', extracted_base, converted_path)
    convert_dicom_segs('CT_LYMPH_NODES', converted_path)
    match_and_save_niftis('CT_LYMPH_NODES', converted_path, 'dicom_seg', None)


def convert_dicom_segs(name, converted_path):
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el)
                   for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    worker = partial(process_dicom_seg)

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker, nifti_paths), total=len(nifti_paths),
                  desc=f"🖼️  Converting {name} DICOM_SEG files"))


def process_dicom_seg(path):
    try:

        img_file = [os.path.join(path, el) for el in os.listdir(path) if el.endswith('.nii.gz')][0]
        seg_src = os.path.join(os.path.join('datasets', 'CT', 'CT_LYMPH_NODES', 'extracted', 'MED_ABD_LYMPH_MASKS', os.path.basename(path), f'{os.path.basename(path)}_mask.nii.gz'))
        shutil.copy(seg_src, os.path.join(path, 'dicom_seg_0.nii.gz'))
        seg = sitk.ReadImage(os.path.join(path, 'dicom_seg_0.nii.gz'))
        seg_binary = sitk.BinaryThreshold(seg, lowerThreshold=1, upperThreshold=65535, insideValue=1, outsideValue=0)
        sitk.WriteImage(seg_binary, os.path.join(path, 'dicom_seg_0.nii.gz'))    
        log_file = os.path.join(path, 'log.txt')
        with open(log_file, 'a+') as f:
            f.write(f"{seg_src} -> {os.path.join(path, f'dicom_seg_0.nii.gz')}\n")

    except Exception as e:
        print(f"❌ Failed to process DICOM SEG {path}: {e}")
        return None




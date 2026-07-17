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

def convert_colorectal_liver_metastases():
    converted_path = os.path.join('datasets', 'CT', 'COLORECTAL_LIVER_METASTASES', 'converted')
    label_classes = ['liver', 'liver_remnant', 'liver_vein', 'portal_vein', 'liver_tumor']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join('datasets', 'CT', 'COLORECTAL_LIVER_METASTASES', 'compressed')
    extracted_base = os.path.join('datasets', 'CT', 'COLORECTAL_LIVER_METASTASES', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('COLORECTAL_LIVER_METASTASES', extracted_base, converted_path)
    convert_dicom_segs('COLORECTAL_LIVER_METASTASES', converted_path)

    match_and_save_niftis('COLORECTAL_LIVER_METASTASES', converted_path, 'dicom_seg', None)


def convert_dicom_segs(name, converted_path):
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el)
                   for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    worker = partial(process_dicom_seg)

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker, nifti_paths), total=len(nifti_paths),
                  desc=f"🖼️  Converting {name} DICOM_SEG files"))


def process_dicom_seg(path):
    try:
        if len([el for el in os.listdir(path) if el.endswith('.dcm')]) == 0:
            shutil.rmtree(path)
            return None
        # Get the DICOM SEG file
        dicom_seg_file = next(
            os.path.join(path, el)
            for el in os.listdir(path)
            if el.endswith('.dcm')
        )
        img_file = [os.path.join(path, el) for el in os.listdir(path) if el.endswith('.nii.gz')][0]

        cmd = [
            "segimage2itkimage",
            "--inputDICOM", dicom_seg_file,
            "--outputDirectory", os.path.dirname(dicom_seg_file),
            "-t", "nii"
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        nii_files = [el for el in os.listdir(path) if el.endswith('.nii.gz')]
        for i in range(1, len(nii_files)):
            if os.path.exists(os.path.join(path, f'{i}.nii.gz')):
                shutil.move(os.path.join(path, f'{i}.nii.gz'), os.path.join(path, f'dicom_seg_{i-1}.nii.gz'))
        tumor = sitk.ReadImage(os.path.join(path, 'dicom_seg_4.nii.gz'))
        for i in range(6, len(nii_files)):
            if os.path.exists(os.path.join(path, f'dicom_seg_{i-1}.nii.gz')):
                tumor = sitk.Or(tumor, sitk.ReadImage(os.path.join(path, f'dicom_seg_{i-1}.nii.gz')))
                os.remove(os.path.join(path, f'dicom_seg_{i-1}.nii.gz'))

        log_file = os.path.join(path, 'log.txt')
        with open(log_file, 'r') as f:
            dicom_source = [el.split('->')[0].rstrip() for el in f.readlines() if dicom_seg_file in el][0]
        with open(log_file, 'a+') as f:
            for i in range(1, 5):
                resample_seg_to_image(os.path.join(path, f'dicom_seg_{i-1}.nii.gz'), img_file, os.path.join(path, f'dicom_seg_{i-1}.nii.gz'))
                f.write(f"{dicom_source} -> {os.path.join(path, f'dicom_seg_{i-1}.nii.gz')}\n")

    except Exception as e:
        print(f"❌ Failed to process DICOM SEG {path}: {e}")
        return None

import SimpleITK as sitk

def resample_seg_to_image(seg_path, image_path, output_path):
    """
    Resample a segmentation NIfTI to exactly match the shape, spacing, origin, and direction of the image.

    Args:
        seg_path (str): Path to segmentation NIfTI.
        image_path (str): Path to reference image NIfTI.
        output_path (str): Path to save resampled segmentation.
    """
    # Read image and segmentation
    img = sitk.ReadImage(image_path)
    seg = sitk.ReadImage(seg_path)

    # Resample segmentation to image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # preserve labels
    resampler.SetDefaultPixelValue(0)                     # background label
    resampler.SetOutputSpacing(img.GetSpacing())
    resampler.SetSize(img.GetSize())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    seg_resampled = resampler.Execute(seg)
    # Make binary: 0 for background, 1 for any label
    seg_binary = sitk.BinaryThreshold(seg_resampled, lowerThreshold=1, upperThreshold=65535, insideValue=1, outsideValue=0)

    # Save output
    sitk.WriteImage(seg_binary, output_path)


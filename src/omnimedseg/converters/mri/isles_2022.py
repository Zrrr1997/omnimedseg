import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *


def convert_isles_2022():
    converted_path = os.path.join('datasets', 'MRI', 'ISLES_2022', 'converted')

    # Create label mapping
    label_classes = ['stroke_lesion']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'ISLES_2022', 'extracted', 'ISLES-2022')
    
    image_paths = []
    image_paths += [os.path.join(extracted_base, f, f'ses-0001', 'anat', f'sub-strokecase{f[-4:]}_ses-0001_FLAIR.nii.gz') for f in os.listdir(extracted_base) if 'sub-strokecase' in f]
    image_paths += [os.path.join(extracted_base, f, f'ses-0001', 'dwi', f'sub-strokecase{f[-4:]}_ses-0001_adc.nii.gz') for f in os.listdir(extracted_base) if 'sub-strokecase' in f]
    image_paths += [os.path.join(extracted_base, f, f'ses-0001', 'dwi', f'sub-strokecase{f[-4:]}_ses-0001_dwi.nii.gz') for f in os.listdir(extracted_base) if 'sub-strokecase' in f]


    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        shutil.copy(file, dest_image_path)

        case_id = file.split('/')[-4][-4:]
        label_file = os.path.join(extracted_base, 'derivatives', f'sub-strokecase{case_id}', f'ses-0001', f'sub-strokecase{case_id}_ses-0001_msk.nii.gz')
        if 'FLAIR' in file:
            resample_seg_to_image(label_file, file, os.path.join(converted_path, 'labels', 'stroke_lesion', filename))
        else:
            shutil.copy(label_file, os.path.join(converted_path, 'labels', 'stroke_lesion', filename))

        return filename, {'image': file, 'stroke_lesion': label_file}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🖼️  Processing ISLES_2022 images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict

    save_mapping(converted_path, filename_map)

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
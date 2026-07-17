import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from src.utils.dataset_utils import *
from src.converters.utils_ct import *
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import cv2
from tqdm import tqdm


def process_image(file, extracted_base, converted_path):
    process_dicom_case(file, 'CHAOS_CT', extracted_base, converted_path)

def process_nifti(file, idx, converted_path, filename_map):
    filename = f"{idx:06d}.nii.gz"
    with open(os.path.join(os.path.dirname(file), 'log.txt'), 'r') as f:
        lines = f.readlines()
        image_source = [el.rstrip().split('->')[0].rstrip() for el in lines if file in el][0]
        label_source = image_source.replace('DICOM_anon', 'Ground')
    filename_map[filename] = {
        'image': image_source,
        'liver': label_source
    }
    pngs_to_nifti(label_source, file, os.path.join(converted_path, 'labels', 'liver', filename))

    shutil.copy(file, os.path.join(converted_path, 'images', filename))

def convert_chaos_ct():
    converted_path = os.path.join('datasets', 'CT', 'CHAOS_CT', 'converted')
    label_classes = ['liver']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'CT', 'CHAOS_CT', 'extracted', 'Train_Sets', 'CT')
    image_paths = [os.path.join(extracted_base, f, 'DICOM_anon') for f in os.listdir(extracted_base)]

    filename_map = {}

    # Parallelize processing of DICOM cases
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda file: process_image(file, extracted_base, converted_path), image_paths), desc="🖼️  Processing CHAOS_CT images", total=len(image_paths)))

    nifti_files = [os.path.join(converted_path, 'all_niftis', f) for f in os.listdir(os.path.join(converted_path, 'all_niftis')) if 'nii.gz' in f]

    # Parallelize processing of NIfTI files
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda args: process_nifti(*args), [(file, idx, converted_path, filename_map) for idx, file in enumerate(nifti_files)]), desc="🖼️  Processing CHAOS_CT images", total=len(nifti_files)))

    save_mapping(converted_path, filename_map)

def load_png_stack(png_dir, num_slices, target_shape):
    """
    Loads PNGs from a directory and stacks them into a 3D numpy array.
    Assumes each PNG is a binary mask with the same spatial resolution as the NIfTI slices.
    """
    slices = []
    png_files = sorted(os.listdir(png_dir))[::-1]
    if len(png_files) != num_slices:
        print('different number of slices than in the image!', png_dir, len(png_files), num_slices)
        return np.zeros(target_shape).astype(np.uint8)
    for i in range(num_slices):
        png_path = os.path.join(png_dir, png_files[i])
        if not os.path.exists(png_path):
            raise FileNotFoundError(f"Missing slice: {png_path}")

        arr = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if 'CHAOS_CT' in png_dir:
            arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)


        if arr.shape != target_shape:
            raise ValueError(f"Shape mismatch on slice {i}: got {arr.shape}, expected {target_shape}")

        binary_arr = (arr > 0).astype(np.uint8)  # Threshold to binary
        slices.append(binary_arr)

    return np.stack(slices, axis=-1)  # Stack along z-axis


def pngs_to_nifti(png_dir, reference_nii_path, output_path):
    # Load reference NIfTI file
    ref_nii = nib.load(reference_nii_path)
    ref_shape = ref_nii.shape[:2]  # ignore number of slices for now
    num_slices = ref_nii.shape[2]

    # Load and stack PNGs
    label_volume = load_png_stack(png_dir, num_slices, ref_shape)
    if np.sum(label_volume) == 0:
        return False
    final_affine = ref_nii.affine

    # Create new NIfTI image
    label_nii = nib.Nifti1Image(label_volume, affine=final_affine, header=ref_nii.header)

    # Save result
    nib.save(label_nii, output_path)
    return True
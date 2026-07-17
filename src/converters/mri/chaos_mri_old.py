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
        label_source = image_source.replace('DICOM_anon', 'Ground').replace('InPhase', '').replace('OutPhase', '')
    filename_map[filename] = {
        'image': image_source,
        'liver': label_source,
        'spleen': label_source,
        'left_kidney': label_source,
        'right_kidney': label_source
    }

    converted_correctly = pngs_to_nifti(label_source, file, os.path.join(converted_path, 'labels', 'liver', filename))
    if converted_correctly:
        shutil.copy(file, os.path.join(converted_path, 'images', filename))

def convert_chaos_mri():
    converted_path = os.path.join('datasets', 'MRI', 'CHAOS_MRI', 'converted')
    label_classes = ['liver', 'right_kidney', 'left_kidney', 'spleen']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', 'MRI', 'CHAOS_MRI', 'extracted', 'Train_Sets', 'MR')
    image_paths = []
    image_paths += [os.path.join(extracted_base, f, 'T1DUAL', 'DICOM_anon', 'InPhase') for f in os.listdir(extracted_base)]
    image_paths += [os.path.join(extracted_base, f, 'T1DUAL', 'DICOM_anon', 'OutPhase') for f in os.listdir(extracted_base)]
    image_paths += [os.path.join(extracted_base, f, 'T2SPIR', 'DICOM_anon') for f in os.listdir(extracted_base)]

    filename_map = {}

    # Parallelize processing of DICOM cases
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda file: process_image(file, extracted_base, converted_path), image_paths), desc="🖼️  Processing CHAOS_MRI images", total=len(image_paths)))

    nifti_files = [os.path.join(converted_path, 'all_niftis', f) for f in os.listdir(os.path.join(converted_path, 'all_niftis')) if 'nii.gz' in f]

    # Parallelize processing of NIfTI files
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda args: process_nifti(*args), [(file, idx, converted_path, filename_map) for idx, file in enumerate(nifti_files)]), desc="🖼️  Processing CHAOS_MRI images", total=len(nifti_files)))

    save_mapping(converted_path, filename_map)

def load_png_stack(png_dir, num_slices, target_shape):
    """
    Loads PNGs from a directory and stacks them into a 3D numpy array.
    Assumes each PNG is a binary mask with the same spatial resolution as the NIfTI slices.
    """
    png_files = sorted(os.listdir(png_dir))
    vols = []

    if len(png_files) != num_slices:
        #print('different number of slices than in the image!', png_dir, len(png_files), num_slices)
        for i in range(4):
            vols.append(np.zeros(target_shape).astype(np.uint8))
        return vols
    for val in [63, 126, 189, 252]:
        slices = []

        for i in range(num_slices):
            png_path = os.path.join(png_dir, png_files[i])
            if not os.path.exists(png_path):
                raise FileNotFoundError(f"Missing slice: {png_path}")

            arr = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if 'CHAOS_MRI' in png_dir:
                arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)


            if arr.shape != target_shape:
                arr = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


            binary_arr = (arr == val).astype(np.uint8)  # Threshold to binary
            slices.append(binary_arr)
        vols.append(np.stack(slices, axis=-1))

    return vols


def pngs_to_nifti(png_dir, reference_nii_path, output_path):
    # Load reference NIfTI file
    ref_nii = nib.load(reference_nii_path)
    ref_shape = ref_nii.shape[:2]  # ignore number of slices for now
    num_slices = ref_nii.shape[2]
    label_classes = ['liver', 'right_kidney', 'left_kidney', 'spleen']

    # Load and stack PNGs
    label_volumes = load_png_stack(png_dir, num_slices, ref_shape)
    if np.sum(label_volumes[0]) == 0:
        return False
    final_affine = ref_nii.affine

    # Create new NIfTI image
    for i, label_volume in enumerate(label_volumes):

        output_path_ = output_path.replace('liver', label_classes[i])
        label_nii = nib.Nifti1Image(label_volume, affine=final_affine, header=ref_nii.header)

        # Save result
        nib.save(label_nii, output_path_)
    return True
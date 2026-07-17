import os
import numpy as np
import pydicom
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import shutil
from functools import partial
from src.utils.dataset_utils import *
from src.converters.utils_ct import *


def get_dicom_spacing(dicom_folder):
    """
    Extract pixel spacing and slice thickness from DICOM files.
    Returns: (pixel_spacing_y, pixel_spacing_x, slice_thickness)
    """
    # Get first DICOM file to extract spacing
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    if not dicom_files:
        return (1.0, 1.0, 1.0)
    
    first_dicom = os.path.join(dicom_folder, dicom_files[0])
    ds = pydicom.dcmread(first_dicom)
    
    # Get pixel spacing (y, x)
    if hasattr(ds, 'PixelSpacing'):
        pixel_spacing_y = float(ds.PixelSpacing[0])
        pixel_spacing_x = float(ds.PixelSpacing[1])
    else:
        pixel_spacing_y = 1.0
        pixel_spacing_x = 1.0
    
    # Get slice thickness
    if hasattr(ds, 'SliceThickness'):
        slice_thickness = float(ds.SliceThickness)
    elif hasattr(ds, 'SpacingBetweenSlices'):
        slice_thickness = float(ds.SpacingBetweenSlices)
    else:
        slice_thickness = 1.0
    
    return (pixel_spacing_y, pixel_spacing_x, slice_thickness)


def load_dicom_series_to_volume(dicom_folder):
    """
    Load all DICOM files in a folder and stack them into a 3D volume.
    Returns: volume (z, y, x) and the sorted DICOM files (for reference)
    """
    # Get all DICOM files
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    dicom_paths = [os.path.join(dicom_folder, f) for f in dicom_files]
    
    # Read all DICOMs to get InstanceNumber for sorting
    dicom_data = []
    for path in dicom_paths:
        ds = pydicom.dcmread(path)
        # Get pixel array
        image = ds.pixel_array.astype(np.float32)

        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # Apply RescaleSlope and RescaleIntercept if present
        if hasattr(ds, 'RescaleSlope'):
            image = image * ds.RescaleSlope
        if hasattr(ds, 'RescaleIntercept'):
            image = image + ds.RescaleIntercept
            
        # Store with InstanceNumber for sorting
        instance_num = getattr(ds, 'InstanceNumber', int(os.path.basename(path).split('-')[-1].replace('.dcm', '')))
        dicom_data.append((instance_num, image))
    
    # Sort by InstanceNumber
    dicom_data.sort(key=lambda x: x[0])
    
    # Stack into volume
    volume = np.stack([img for _, img in dicom_data], axis=2)
    spacing = get_dicom_spacing(dicom_folder)
    return volume, spacing


def load_png_labels_to_volume(png_folder, num_slices, target_spatial_shape):
    """
    Load PNG label files and create a multi-label volume.
    For CHAOS: different pixel values represent different organs
    Value mapping:
    63 = liver, 126 = right kidney, 189 = left kidney, 252 = spleen
    """
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
    
    if len(png_files) != num_slices:
        # Pad or truncate to match
        if len(png_files) < num_slices:
            # Pad with zeros
            png_files.extend([''] * (num_slices - len(png_files)))
        else:
            # Truncate
            png_files = png_files[:num_slices]
    
    # Initialize volume (y, x, z) - note orientation
    label_volume = np.zeros((*target_spatial_shape, num_slices), dtype=np.uint8)
    
    for i, png_file in enumerate(png_files):
        if not png_file:
            continue
            
        png_path = os.path.join(png_folder, png_file)
        arr = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        
        # CHAOS MRI PNGs need 90-degree clockwise rotation
        arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        
        # Resize if needed
        if arr.shape != target_spatial_shape:
            arr = cv2.resize(arr, (target_spatial_shape[1], target_spatial_shape[0]), 
                           interpolation=cv2.INTER_NEAREST)
        
        label_volume[:, :, i] = arr
    
    return label_volume


def process_single_case(case_id, base_dir, output_dir, idx):
    """
    Process a single case: convert DICOM and PNG to volumes, then save as NIfTI.
    """
    data_dict = {}
    case_path = os.path.join(base_dir, str(case_id))
    filename = f"{idx:06d}.nii.gz"

    for modality in ['InPhase', 'OutPhase']:
        if modality == 'OutPhase':
            filename = f"{idx+1:06d}.nii.gz"

        dicom_folder = os.path.join(case_path, 'T1DUAL', 'DICOM_anon', modality)
        png_folder = os.path.join(case_path, 'T1DUAL', 'Ground')
        
        if os.path.exists(dicom_folder) and os.path.exists(png_folder):
            
            # Load DICOM volume
            image_volume, spacing = load_dicom_series_to_volume(dicom_folder)
            
            # Load PNG labels
            label_volume = load_png_labels_to_volume(
                png_folder, 
                num_slices=image_volume.shape[0],
                target_spatial_shape=image_volume.shape[0:2]
            )
            
            # Save as NIfTI
            affine = np.diag([1, 1, 1, 1])  # Identity affine, adjust if you have spacing info
            affine[0, 0] = spacing[0]  # pixel_spacing_x
            affine[1, 1] = spacing[1]  # pixel_spacing_y
            affine[2, 2] = spacing[2]  # slice_thickness
            
            # Save image
            img_output_path = os.path.join(output_dir, 'images', filename)
            os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
            img_nifti = nib.Nifti1Image(image_volume, affine=affine)
            nib.save(img_nifti, img_output_path)
            
            # Save labels
            for i, label in enumerate(['liver', 'right_kidney', 'left_kidney', 'spleen']):
                label_output_path = os.path.join(output_dir, 'labels', label, filename)
                os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
                label_nifti = nib.Nifti1Image((label_volume==63*(i+1)).astype(np.uint8), affine=affine)
                nib.save(label_nifti, label_output_path)
            data_dict[filename] = {
                'image': dicom_folder,
                'liver': png_folder,
                'right_kidney': png_folder,
                'left_kidney': png_folder,
                'spleen': png_folder
            }
            
    dicom_folder = os.path.join(case_path, 'T2SPIR', 'DICOM_anon')
    png_folder = os.path.join(case_path, 'T2SPIR', 'Ground')
    
    if os.path.exists(dicom_folder) and os.path.exists(png_folder):
        filename = f"{idx+2:06d}.nii.gz"

        # Load DICOM volume
        image_volume, spacing = load_dicom_series_to_volume(dicom_folder)
        
        # Load PNG labels
        label_volume = load_png_labels_to_volume(
            png_folder, 
            num_slices=image_volume.shape[0],
            target_spatial_shape=image_volume.shape[0:2]
        )
        
        # Save as NIfTI
        affine = np.diag([1, 1, 1, 1])  # Identity affine, adjust if you have spacing info
        affine[0, 0] = spacing[0]  # pixel_spacing_x
        affine[1, 1] = spacing[1]  # pixel_spacing_y
        affine[2, 2] = spacing[2]  # slice_thickness

        # Save image
        img_output_path = os.path.join(output_dir, 'images', filename)
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        img_nifti = nib.Nifti1Image(image_volume, affine=affine)
        nib.save(img_nifti, img_output_path)
        
        # Save labels
        for i, label in enumerate(['liver', 'right_kidney', 'left_kidney', 'spleen']):
            label_output_path = os.path.join(output_dir, 'labels', label, filename)
            os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
            label_nifti = nib.Nifti1Image((label_volume==63*(i+1)).astype(np.uint8), affine=affine)
            nib.save(label_nifti, label_output_path)
        data_dict[filename] = {
            'image': dicom_folder,
            'liver': png_folder,
            'right_kidney': png_folder,
            'left_kidney': png_folder,
            'spleen': png_folder
        }
    return data_dict, idx + 3


def process_case_wrapper(args):
    """Wrapper function for multiprocessing to unpack arguments."""
    case_id, base_dir, output_dir, start_idx = args
    return process_single_case(case_id, base_dir, output_dir, start_idx)


def convert_chaos_mri(max_workers=None):
    """
    Main function to convert CHAOS MRI dataset directly from DICOM/PNG to NIfTI.
    
    Args:
        max_workers: Number of parallel processes (default: CPU count)
    """
    
    base_dir = os.path.join('datasets', 'MRI', 'CHAOS_MRI', 'extracted', 'Train_Sets', 'MR')
    output_dir = os.path.join('datasets', 'MRI', 'CHAOS_MRI', 'converted')
    filename_map = {}
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Get all case IDs
    case_ids = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.isdigit():
            case_ids.append(int(item))
    
    case_ids.sort()
    print(f"Found {len(case_ids)} cases: {case_ids}")
    
    # Calculate starting indices for each case
    # Each case produces 3 files (InPhase, OutPhase, T2SPIR)
    case_indices = [(case_id, base_dir, output_dir, i * 3) for i, case_id in enumerate(case_ids)]
    
    # Process cases in parallel using ProcessPoolExecutor
    # Use ProcessPoolExecutor for CPU-bound operations
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map with tqdm for progress bar
        results = list(tqdm(
            executor.map(process_case_wrapper, case_indices),
            total=len(case_ids),
            desc="Processing cases"
        ))
    
    # Collect results
    for data_dict, _ in results:
        filename_map.update(data_dict)
    
    save_mapping(output_dir, filename_map)
    
    print(f"Conversion complete! Processed {len(case_ids)} cases.")
    print(f"Output saved to: {output_dir}")


# Alternative version using ThreadPoolExecutor for I/O-heavy operations
def convert_chaos_mri_threaded(max_workers=8):
    """
    Threaded version for I/O-bound operations (if network/disk is bottleneck).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    base_dir = os.path.join('datasets', 'MRI', 'CHAOS_MRI', 'extracted', 'Train_Sets', 'MR')
    output_dir = os.path.join('datasets', 'MRI', 'CHAOS_MRI', 'converted')
    filename_map = {}
    map_lock = threading.Lock()
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Get all case IDs
    case_ids = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.isdigit():
            case_ids.append(int(item))
    
    case_ids.sort()
    print(f"Found {len(case_ids)} cases: {case_ids}")
    
    def process_and_update(case_id, idx):
        data_dict, _ = process_single_case(case_id, base_dir, output_dir, idx)
        with map_lock:
            filename_map.update(data_dict)
        return data_dict
    
    # Process cases in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, case_id in enumerate(case_ids):
            future = executor.submit(process_and_update, case_id, i * 3)
            futures.append(future)
        
        # Monitor progress
        for future in tqdm(futures, desc="Processing cases"):
            future.result()  # Wait for completion
    
    save_mapping(output_dir, filename_map)
    print(f"Conversion complete! Processed {len(case_ids)} cases.")



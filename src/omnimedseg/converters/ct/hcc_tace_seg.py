import os
import numpy as np
import nibabel as nib
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import nibabel as nib
from nibabel.orientations import (
    axcodes2ornt, ornt_transform, io_orientation, inv_ornt_aff, apply_orientation, aff2axcodes
)

def convert_hcc_tace_seg():
    converted_path = os.path.join('datasets', 'CT', 'HCC_TACE_SEG', 'converted')
    label_classes = ['liver', 'liver_tumor', 'liver_vessel', 'aorta']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join('datasets', 'CT', 'HCC_TACE_SEG', 'compressed')
    extracted_base = os.path.join('datasets', 'CT', 'HCC_TACE_SEG', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('HCC_TACE_SEG', extracted_base, converted_path)
    convert_dicom_segs('HCC_TACE_SEG', converted_path)

    match_and_save_niftis('HCC_TACE_SEG', converted_path, 'dicom_seg', None)


def convert_dicom_segs(name, converted_path):
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el)
                   for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    worker = partial(process_dicom_seg)

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker, nifti_paths), total=len(nifti_paths),
                  desc=f"🖼️  Converting {name} DICOM_SEG files"))


def process_dicom_seg(path):
    try:
        # Get the DICOM SEG file
        dicom_seg_file = next(
            os.path.join(path, el)
            for el in os.listdir(path)
            if el.endswith('.dcm')
        )
        df = pd.read_excel(os.path.join('datasets', 'CT', 'HCC_TACE_SEG', 'HCC-TACE-Seg_v1_202201-nbia-digest.xlsx'), engine="openpyxl")

        cmd = [
            "segimage2itkimage",
            "--inputDICOM", dicom_seg_file,
            "--outputDirectory", os.path.dirname(dicom_seg_file),
            "-t", "nii"
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


        log_file = os.path.join(path, 'log.txt')
        with open(log_file, 'r') as f:
            lines = f.readlines()   
        dcm_source_series = [line for line in lines if os.path.basename(dicom_seg_file) in line][0].split('->')[0].strip().split('/')[-1]
        patient_id = os.path.basename(path)

        filtered = df[
            (df['Patient ID'] == patient_id)
        ]['Series Instance UID'].values
        filtered = [el for el in filtered if el != dcm_source_series]

        for image_series in filtered:
            dicom_dir = os.path.join('datasets', 'CT', 'HCC_TACE_SEG', 'extracted', image_series, image_series)
            try:
                old_files = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith('.nii.gz') 
                ]
                subprocess.run([
                    "dcm2niix", "-z", "y", "-x", "n", "-o", path, "-f", "%p_%s", dicom_dir
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                new_files = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith('.nii.gz') if os.path.join(path, f) not in old_files
                ]
                for new_file in new_files:
                    with open(log_file, 'a+') as f:
                        f.write(f"{dicom_dir} -> {new_file}\n")

            except subprocess.CalledProcessError:
                try:
                    reader = sitk.ImageSeriesReader()
                    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
                    if not series_ids:
                        raise RuntimeError("No DICOM series found.")
                    series_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
                    reader.SetFileNames(series_files)
                    image = reader.Execute()
                    patient_id = print_patient_id(dicom_dir=dicom_dir)
                    output_file = os.path.join(path, f"{patient_id}.nii.gz")
                    sitk.WriteImage(image, output_file)
                except Exception as e:
                    print(f"❌ Failed to convert {dicom_dir} with both dcm2niix and SimpleITK: {e}")
                    return

        nifti_files = [
            os.path.join(path, el)
            for el in os.listdir(path)
            if el.endswith('.nii.gz')]

        for file in nifti_files:
            if len(os.path.basename(file)) <= 10: # Label file, e.g. 1.nii.gz
                continue

            label_img = nib.load(os.path.join(path, '1.nii.gz'))
            image_img = nib.load(file)

            label_data = label_img.get_fdata()
            image_data = image_img.get_fdata()

            if image_data.ndim == 4:
                # Only take the first volume along the last axis
                image_data = image_data[:, :, :, 0]

            if label_data.shape != image_data.shape or remove_condition(file):
                os.remove(file)
                continue
            else:
                if patient_id in ['HCC_003', 'HCC_002', 'HCC_007', 'HCC_030', 'HCC_093', 'HCC_104', 'HCC_105']:
                    if patient_id not in ['HCC_011']:
                        # Flip along axes
                        data = np.flip(image_data, axis=1)
                        if patient_id in ['HCC_003']:
                            data = np.flip(image_data, axis=0)
                    # Preserve metadata: affine + header
                    new_img = nib.Nifti1Image(data, affine=label_img.affine, header=label_img.header)

                    # Save
                    nib.save(new_img, file)

        for i in np.arange(1, 5):
            shutil.move(os.path.join(path, f'{i}.nii.gz'), os.path.join(path, f'dicom_seg_{i}.nii.gz'))

        nifti_image_files = [
            os.path.join(path, el)
            for el in os.listdir(path)
            if el.endswith('.nii.gz') and 'dicom_seg' not in el]

        if len(nifti_image_files) > 1:
            for i, nii in enumerate(nifti_image_files[1:]):
                new_path = path.replace(os.path.basename(path), f'{os.path.basename(path)}_{i+1}')
                os.makedirs(new_path, exist_ok=True)
                shutil.move(nii, os.path.join(new_path, os.path.basename(nii)))
                shutil.copy(os.path.join(path, 'log.txt'), os.path.join(new_path, 'log.txt'))
                for i in np.arange(1, 5):
                    shutil.copy(os.path.join(path, f'dicom_seg_{i}.nii.gz'), os.path.join(new_path, f'dicom_seg_{i}.nii.gz'))
        if len(nifti_image_files) == 0:
            shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"❌ Failed to process DICOM SEG {path}: {e}")
        return None

def remove_condition(file):
    conditions = [
        ('HCC_005', 'Recon'),
        ('HCC_009', 'Recon'),
        ('HCC_014', 'Recon'),
        ('HCC_019', 'Recon'),
        ('HCC_039', 'Recon'),
        ('HCC_013', '_4a'),
        ('HCC_017', '6.3_@_ABD_(LIVER_3_PHASE)_5CC_@_5', False),
        ('HCC_025', 'Recon_3'),
        ('HCC_027', '_102'),
        ('HCC_034', 'Recon_3_LIVER_3_PHASE_(AP)_5', False),
        ('HCC_040', 'Recon_3_LIVER_3_PHASE_(C_A_P)_5', False),
        ('HCC_042', 'Recon_3_LIVER_3_PHASE_(AP)_7', False),
        ('HCC_060', '_6a', False),
        ('HCC_085', None),
        ('HCC_011', None),
        ('HCC_096', None),
        ('HCC_100', None),
        ('HCC_104', None),
    ]

    no_recon_2 = {
        'HCC_020', 'HCC_022', 'HCC_023', 'HCC_026', 'HCC_029', 'HCC_032', 'HCC_033', 'HCC_035', 'HCC_038',
        'HCC_041', 'HCC_045', 'HCC_046', 'HCC_049', 'HCC_050', 'HCC_051', 'HCC_053', 'HCC_055', 'HCC_056',
        'HCC_058', 'HCC_059', 'HCC_061', 'HCC_062', 'HCC_066', 'HCC_067', 'HCC_070', 'HCC_071', 'HCC_072',
        'HCC_073', 'HCC_076', 'HCC_078', 'HCC_079', 'HCC_081', 'HCC_083', 'HCC_084', 'HCC_086', 'HCC_087',
        'HCC_098'
    }

    for condition in conditions:
        if condition[0] in file and (condition[1] is None or (condition[1] in file) == (len(condition) < 3 or condition[2])):
            return True

    if any(el in file and 'Recon_2' in file for el in no_recon_2):
        return True

    return False

def spine_is_on_top(mid_slice_2d: np.ndarray) -> bool:
    """Detect if spine is on top half of axial slice."""
    # make sure the slice is 2D
    sl = np.squeeze(mid_slice_2d).astype(np.float32)
    
    if sl.ndim != 2:
        # fallback: take first 2D plane if extra dim exists
        sl = sl[:, :, 0] if sl.ndim == 3 else sl

    # normalize
    p1, p99 = np.percentile(sl, [1, 99])
    if p99 <= p1:
        return False
    sl = np.clip((sl - p1) / (p99 - p1), 0, 1)

    h, w = sl.shape
    c0, c1 = int(0.33 * w), int(0.67 * w)
    roi = sl[:, c0:c1]
    thr = np.percentile(roi, 98)
    mask = roi >= thr

    return mask[:h//2].sum() > mask[h//2:].sum()
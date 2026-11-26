from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import nibabel as nib

def convert_cc_tumor_heterogeneity():
    converted_path = os.path.join(data_root(), 'MRI', 'CC_TUMOR_HETEROGENEITY', 'converted')
    label_classes = ['cervical_tumor']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'CC_TUMOR_HETEROGENEITY', 'extracted')
    compressed_base = os.path.join(data_root(), 'MRI', 'CC_TUMOR_HETEROGENEITY', 'compressed')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('CC_TUMOR_HETEROGENEITY', extracted_base, converted_path)
    convert_rtstructs('CC_TUMOR_HETEROGENEITY', converted_path, 'CT-SEG')

    match_and_save_niftis_cc_tumor()


def match_and_save_niftis_cc_tumor():
    converted_path = os.path.join(data_root(), 'MRI', 'CC_TUMOR_HETEROGENEITY', 'converted')

    patients = [os.path.join(converted_path, 'all_niftis', el) for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]
    idx = 0
    image_dest = os.path.join(converted_path, 'images')
    filename_map = {}
    for patient in patients:
        with open(os.path.join(patient, 'log.txt'), 'r') as f:
            lines = f.readlines()
        for i in range(3):
            if os.path.exists(os.path.join(patient, f'{i}')):
                nii_files = [os.path.join(patient, f'{i}', el) for el in os.listdir(os.path.join(patient, f'{i}')) if 'Struct' not in el]
                rt_file = [os.path.join(patient, f'{i}', el) for el in os.listdir(os.path.join(patient, f'{i}')) if 'Struct' in el][0]
                rt_data = nib.load(rt_file).get_fdata()


                    
                for nii_file in nii_files:
                    img_data = nib.load(nii_file).get_fdata()
                    if img_data.shape != rt_data.shape:
                        continue
                    if 'B08' in nii_file and '4a.nii' in nii_file:
                        continue
                    if 'B10' in nii_file and 'Parallel_5a' in nii_file:
                        continue
                    if 'A10' in nii_file and '700a' in nii_file:
                        continue
                    if 'A03' in nii_file and '8118_4b' in nii_file:
                        continue
                    if 'A08' in nii_file and '_700' in nii_file:
                        continue

                    else:
                        filename = f"{idx:06d}.nii.gz"
                        shutil.copy(nii_file, os.path.join(image_dest, filename))
                        shutil.copy(rt_file, os.path.join(converted_path, 'labels', 'cervical_tumor', filename))
                        image_source = [el.rstrip().split('->')[0] for el in lines if os.path.basename(nii_file) in el][0]
                        label_source = [el.rstrip().split('->')[0] for el in lines if os.path.basename(rt_file) in el][0]

                        filename_map[filename] = {'image': image_source, 'cervical_tumor': label_source}
                        idx += 1
    save_mapping(converted_path, filename_map)
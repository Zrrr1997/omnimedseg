from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import SimpleITK as sitk
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def convert_plethora():
    converted_path = os.path.join(data_root(), 'CT', 'PLETHORA', 'converted')
    label_classes = ['left_lung', 'right_lung', 'pleural_effusion', 'pleural_effusion_annotator_1', 'pleural_effusion_annotator_2']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join(data_root(), 'CT', 'PLETHORA', 'compressed')
    extracted_base = os.path.join(data_root(), 'CT', 'PLETHORA', 'extracted')
    #unzip_tcia(extracted_base, compressed_base)
    #convert_extracted_dicoms('PLETHORA', extracted_base, converted_path)
    match_plethora()



def process_patient(patient, idx, nifti_dir, converted_path, extracted_base):
    filename = f"{idx:06d}.nii.gz"
    data_dict = {}
    patient_dir = os.path.join(nifti_dir, patient)

    # Copy image
    nii_img = [os.path.join(patient_dir, el) for el in os.listdir(patient_dir) if '.nii.gz' in el][0]
    log_file = os.path.join(patient_dir, 'log.txt')
    with open(log_file, 'r') as f:
        dicom_source = [el.split('->')[0].rstrip() for el in f.readlines() if nii_img in el][0]
        data_dict['image'] = dicom_source
    shutil.copy(nii_img, os.path.join(converted_path, 'images', filename))

    # Thoracic cavities
    thoracic_dir = os.path.join(extracted_base, 'Thoracic_Cavities', patient)
    if os.path.exists(thoracic_dir):
        label_path = os.path.join(thoracic_dir, f'{patient}_thor_cav_primary_reviewer.nii.gz')
        lung_labels = sitk.ReadImage(label_path)
        left_lung = sitk.Cast(sitk.Equal(lung_labels, 1), sitk.sitkUInt8)
        right_lung = sitk.Cast(sitk.Equal(lung_labels, 2), sitk.sitkUInt8)
        sitk.WriteImage(left_lung, os.path.join(converted_path, 'labels', 'left_lung', filename))
        sitk.WriteImage(right_lung, os.path.join(converted_path, 'labels', 'right_lung', filename))
        data_dict['left_lung'] = label_path
        data_dict['right_lung'] = label_path

    # Pleural Effusion
    effusion_dir = os.path.join(extracted_base, 'Effusions', patient)
    if os.path.exists(effusion_dir):
        label_path_1 = os.path.join(effusion_dir, f'{patient}_effusion_first_reviewer.nii.gz')
        label_path_2 = os.path.join(effusion_dir, f'{patient}_effusion_second_reviewer.nii.gz')
        shutil.copy(label_path_1, os.path.join(converted_path, 'labels', 'pleural_effusion_annotator_1', filename))
        shutil.copy(label_path_2, os.path.join(converted_path, 'labels', 'pleural_effusion_annotator_2', filename))

        eff1 = sitk.ReadImage(label_path_1)
        eff2 = sitk.ReadImage(label_path_2)
        merged_label = sitk.Cast(sitk.Or(eff1 > 0, eff2 > 0), sitk.sitkUInt8)
        merged_out = os.path.join(converted_path, 'labels', 'pleural_effusion', filename)
        sitk.WriteImage(merged_label, merged_out)

        data_dict['pleural_effusion_annotator_1'] = label_path_1
        data_dict['pleural_effusion_annotator_2'] = label_path_2
        data_dict['pleural_effusion'] = merged_out

    return filename, data_dict


def match_plethora():
    nifti_dir = os.path.join(data_root(), 'CT/PLETHORA/converted/all_niftis')
    converted_path = os.path.join(data_root(), 'CT', 'PLETHORA', 'converted')
    extracted_base = os.path.join(data_root(), 'CT', 'PLETHORA', 'extracted')

    # Ensure label directories exist
    for lbl_dir in ['left_lung', 'right_lung', 'pleural_effusion_annotator_1',
                    'pleural_effusion_annotator_2', 'pleural_effusion']:
        os.makedirs(os.path.join(converted_path, 'labels', lbl_dir), exist_ok=True)
    os.makedirs(os.path.join(converted_path, 'images'), exist_ok=True)

    patients = os.listdir(nifti_dir)
    filename_map = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_patient, patient, idx, nifti_dir, converted_path, extracted_base): patient
                   for idx, patient in enumerate(patients)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PLETHORA patients"):
            filename, data_dict = future.result()
            filename_map[filename] = data_dict

    save_mapping(converted_path, filename_map)


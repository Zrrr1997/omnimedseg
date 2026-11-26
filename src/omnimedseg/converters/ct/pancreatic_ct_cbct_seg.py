from omnimedseg.core.paths import data_root
import os
import shutil
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_pancreatic_ct_cbct_seg():
    converted_path = os.path.join(data_root(), 'CT', 'PANCREATIC_CT_CBCT_SEG', 'converted')
    label_classes = ['organs']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join(data_root(), 'CT', 'PANCREATIC_CT_CBCT_SEG', 'compressed')
    extracted_base = os.path.join(data_root(), 'CT', 'PANCREATIC_CT_CBCT_SEG', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('PANCREATIC_CT_CBCT_SEG', extracted_base, converted_path)
    remove_cbct_and_eclipse()
    convert_rtstructs('PANCREATIC_CT_CBCT_SEG', converted_path, 'CT-SEG')

    match_and_save_niftis('PANCREATIC_CT_CBCT_SEG', converted_path, 'Struct', None)


def remove_cbct_and_eclipse():
    for patient in os.listdir(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis')):
        nii_files = [el for el in os.listdir(os.path.join(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis'), patient)) if '.nii.gz' in el]
        if ('CB_016' not in patient) and ('CB_035' not in patient):
            for nii_file in nii_files:
                if 'Eclipse' in nii_file or 'Aligned' in nii_file or nii_file.startswith('_'):
                    os.remove(os.path.join(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis'), patient, nii_file))
                    os.remove(os.path.join(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis'), patient, nii_file.replace('.nii.gz', '.json')))
        elif 'CB_016' in patient:
            for nii_file in nii_files:
                if '_3.nii.gz' not in nii_file:
                    os.remove(os.path.join(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis'), patient, nii_file))
                    os.remove(os.path.join(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis'), patient, nii_file.replace('.nii.gz', '.json')))
        elif 'CB_035' in patient:
            for nii_file in nii_files:
                if 'Aligned_CT_-_test1_44269.nii.gz' not in nii_file:
                    os.remove(os.path.join(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis'), patient, nii_file))
                    os.remove(os.path.join(os.path.join(data_root(), 'CT/PANCREATIC_CT_CBCT_SEG/converted/all_niftis'), patient, nii_file.replace('.nii.gz', '.json')))   


from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_nsclc_radiomics():
    converted_path = os.path.join(data_root(), 'CT', 'NSCLC_RADIOMICS', 'converted')
    label_classes = ['lung_tumor', 'esophagus', 'heart', 'spinal_cord', 'left_lung', 'right_lung', 'lungs']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join(data_root(), 'CT', 'NSCLC_RADIOMICS', 'compressed')
    extracted_base = os.path.join(data_root(), 'CT', 'NSCLC_RADIOMICS', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('NSCLC_RADIOMICS', extracted_base, converted_path)
    convert_rtstructs('NSCLC_RADIOMICS', converted_path, 'Segmentation')

    match_and_save_niftis('NSCLC_RADIOMICS', converted_path, 'Struct', None)
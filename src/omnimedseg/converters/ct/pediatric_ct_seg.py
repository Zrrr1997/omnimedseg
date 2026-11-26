from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_pediatric_ct_seg():
    converted_path = os.path.join(data_root(), 'CT', 'PEDIATRIC_CT_SEG', 'converted')
    label_classes = ['left_adrenal_gland', 'right_adrenal_gland', 'bladder', 'bones', 'left_lung', 'right_lung', 'prostate', 'left_breast', 'right_breast', 'duodenum', 'esophagus', 'left_femoral_head', 'right_femoral_head', 'gallbladder', 'gonads', 'heart', 'left_kidney', 'right_kidney', 'large_intestine', 'liver', 'pancreas', 'rectum', 'skin', 'small_intestine', 'spinal_canal', 'spleen', 'stomach', 'thymus', 'utero_cervix']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join(data_root(), 'CT', 'PEDIATRIC_CT_SEG', 'compressed')
    extracted_base = os.path.join(data_root(), 'CT', 'PEDIATRIC_CT_SEG', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('PEDIATRIC_CT_SEG', extracted_base, converted_path)
    convert_rtstructs('PEDIATRIC_CT_SEG', converted_path, 'CT-SEG')

    match_and_save_niftis('PEDIATRIC_CT_SEG', converted_path, 'Struct', None)
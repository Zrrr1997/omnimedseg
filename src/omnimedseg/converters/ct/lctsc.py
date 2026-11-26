import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_lctsc():
    converted_path = os.path.join('datasets', 'CT', 'LCTSC', 'converted')
    label_classes = ['right_lung', 'left_lung', 'esophagus', 'heart', 'spinal_cord']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join('datasets', 'CT', 'LCTSC', 'compressed')
    extracted_base = os.path.join('datasets', 'CT', 'LCTSC', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('LCTSC', extracted_base, converted_path)
    convert_rtstructs('LCTSC', converted_path, 'CT-SEG')

    match_and_save_niftis('LCTSC', converted_path, 'Struct', None)
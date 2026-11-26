from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_prostate_edge_cases():
    converted_path = os.path.join(data_root(), 'CT', 'PROSTATE_EDGE_CASES', 'converted')
    label_classes = ['prostate', 'rectum', 'bladder', 'left_femoral_head', 'right_femoral_head']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join(data_root(), 'CT', 'PROSTATE_EDGE_CASES', 'compressed')
    extracted_base = os.path.join(data_root(), 'CT', 'PROSTATE_EDGE_CASES', 'extracted')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('PROSTATE_EDGE_CASES', extracted_base, converted_path)
    convert_rtstructs('PROSTATE_EDGE_CASES', converted_path, 'Segmentation')

    match_and_save_niftis('PROSTATE_EDGE_CASES', converted_path, 'Struct', None)
from src.converters.utils_ct import *

def convert_msd_prostate():
    convert_msd('MSD_PROSTATE', ['prostate_peripheral_zone', 'prostate_transitional_zone'], 'Task05_Prostate', modality='MRI')
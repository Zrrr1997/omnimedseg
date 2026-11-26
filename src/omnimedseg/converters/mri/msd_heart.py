from omnimedseg.converters.utils_ct import *

def convert_msd_heart():
    convert_msd('MSD_HEART', 'left_atrium', 'Task02_Heart', modality='MRI')
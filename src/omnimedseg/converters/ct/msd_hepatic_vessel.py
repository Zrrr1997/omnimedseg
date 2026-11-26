from omnimedseg.converters.utils_ct import *

def convert_msd_hepatic_vessel():
    convert_msd('MSD_HEPATIC_VESSEL', ['liver_vessel', 'liver_tumor'], 'Task08_HepaticVessel')
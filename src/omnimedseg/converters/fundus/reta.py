from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_fundus import *

from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_reta():
    # Define paths
    converted_path = os.path.join(data_root(), 'Fundus', 'RETA', 'converted')
    ensure_directories(converted_path, ['vessels'])

    extracted_base = os.path.join(data_root(), 'Fundus', 'RETA', 'extracted', 'images')
    convention_map = {'image': 'img', 'vessels': 'vessel'}

    convert_train_test_dataset(extracted_base, converted_path, ['train'], convention_map, label_name_extension='_vessel')




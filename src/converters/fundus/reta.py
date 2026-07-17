import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *
from src.converters.utils_fundus import *

from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_reta():
    # Define paths
    converted_path = os.path.join('datasets', 'Fundus', 'RETA', 'converted')
    ensure_directories(converted_path, ['vessels'])

    extracted_base = os.path.join('datasets', 'Fundus', 'RETA', 'extracted', 'images')
    convention_map = {'image': 'img', 'vessels': 'vessel'}

    convert_train_test_dataset(extracted_base, converted_path, ['train'], convention_map, label_name_extension='_vessel')




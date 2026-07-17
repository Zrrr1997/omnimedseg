import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *
from src.converters.utils_fundus import *

def convert_orvs():
    # Define paths
    converted_path = os.path.join('datasets', 'Fundus', 'ORVS', 'converted')
    ensure_directories(converted_path, ['vessels'])

    extracted_base = os.path.join('datasets', 'Fundus', 'ORVS', 'extracted')
    convention_map = {'image': os.path.join('Original', 'Images'), 'vessels': os.path.join('Original', 'Labels')}

    convert_train_test_dataset(extracted_base, converted_path, ['Train', 'Test'], convention_map)
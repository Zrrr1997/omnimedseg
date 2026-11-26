from omnimedseg.core.paths import data_root
import os 
import cv2
from tqdm import tqdm
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_fundus import *

def convert_fives():
    # Define paths
    converted_path = os.path.join(data_root(), 'Fundus', 'FIVES', 'converted')
    ensure_directories(converted_path, ['vessels'])

    extracted_base = os.path.join(data_root(), 'Fundus', 'FIVES', 'extracted', 'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation')
    convention_map = {'image': 'Original', 'vessels': 'Ground truth'}

    convert_train_test_dataset(extracted_base, converted_path, ['train', 'test'], convention_map)

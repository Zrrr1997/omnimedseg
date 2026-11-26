from omnimedseg.core.paths import data_root
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnimedseg.utils.dataset_utils import *
import h5py
from scipy.io import loadmat
def convert_occisc():
    converted_path = os.path.join(data_root(), 'Microscopy', 'OCCISC', 'converted')
    ensure_directories(converted_path, ['cervical_cell', 'nucleus', 'cytoplasm'])

    extracted_base = os.path.join(data_root(), 'Microscopy', 'OCCISC', 'extracted')
    image_files = []
    image_files = [os.path.join(extracted_base, 'Dataset', 'EDF', f) for f in os.listdir(os.path.join(extracted_base, 'Dataset', 'EDF')) if '_GT' not in f]

    idx = 0  

    filename_map = {}
    split_map = {'train': [], 'test': [], 'unknown': []}

    for file in tqdm(image_files, desc="🖼️  Processing OCCISC images"):
        filename = f"{idx:06d}.png"
        image = cv2.imread(file)
        label = cv2.imread(file.replace('.png', '_GT.png'))
        cell = (label > 0).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus', filename), cell)


        filename_map[filename] = {
            'image': file,
            'nucleus': file.replace('.png', '_GT.png') 
        }
        split_map['unknown'].append(filename)
        idx += 1
    with h5py.File(os.path.join(extracted_base, 'Dataset', 'Synthetic', 'trainset.mat'), 'r') as f:
        trainset_refs = f['trainset']
        synthetic_train = [np.array(f[ref]) for ref in trainset_refs[0]]
        

    synthetic_train_gt = loadmat(os.path.join(extracted_base, 'Dataset', 'Synthetic', 'trainset_GT.mat'))
    synthetic_nuclei = synthetic_train_gt['train_Nuclei'][:,0]
    synthetic_cytoplasm = synthetic_train_gt['train_Cytoplasm'][:,0]


    for i, image in tqdm(enumerate(synthetic_train)):
        filename = f"{idx:06d}.png"

        nucleus = (synthetic_nuclei[i] > 0).astype(np.uint8) * 255
        nucleus_flipped = np.fliplr(nucleus)
        nucleus = np.rot90(nucleus_flipped, k=1)
        cytoplasm = (np.sum(synthetic_cytoplasm[i], axis=0)[0] > 0).astype(np.uint8) * 255
        cytoplasm_flipped = np.fliplr(cytoplasm)
        cytoplasm = np.rot90(cytoplasm_flipped, k=1)
        cell = cytoplasm
        cytoplasm = cytoplasm - nucleus
        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus', filename), nucleus)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'cytoplasm', filename), cytoplasm)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'cervical_cell', filename), cell)

        filename_map[filename] = {
            'image': os.path.join(extracted_base, 'Dataset', 'Synthetic', 'trainset.mat'),
            **{k: os.path.join(extracted_base, 'Dataset', 'Synthetic', 'trainset_GT.mat') for k in ['cervical_cell', 'nucleus', 'cytoplasm']}
        } 
        split_map['train'].append(filename)     
        idx += 1
    
    with h5py.File(os.path.join(extracted_base, 'Dataset', 'Synthetic', 'testset.mat'), 'r') as f:
        testset_refs = f['testset']
        synthetic_test = [np.array(f[ref]) for ref in testset_refs[0]]
    with h5py.File(os.path.join(extracted_base, 'Dataset', 'Synthetic', 'testset_GT.mat'), 'r') as f:
        synthetic_nuclei = f['test_Nuclei']
        synthetic_nuclei = [np.array(f[ref]) for ref in synthetic_nuclei[0]]

        synthetic_cytoplasm_refs = f['test_Cytoplasm']

        for ref in synthetic_cytoplasm_refs:
            # Dereference
            synthetic_cytoplasms = []
            for sub_ref in ref:
                synthetic_cytoplasm = np.array(f[f[sub_ref][0][0]])
                num_refs = int(f[sub_ref].shape[1])
                for i in np.arange(1, num_refs):
                    synthetic_cytoplasm += np.array(f[f[sub_ref][0][i]])
                synthetic_cytoplasms.append(synthetic_cytoplasm)

    for i, image in tqdm(enumerate(synthetic_test)):
        filename = f"{idx:06d}.png"
        nucleus = (synthetic_nuclei[i] > 0).astype(np.uint8) * 255

        cytoplasm = (synthetic_cytoplasms[i] > 0).astype(np.uint8) * 255
        cell = cytoplasm
        cytoplasm = cytoplasm - nucleus
        cv2.imwrite(os.path.join(converted_path, 'images', filename), image)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'nucleus', filename), nucleus)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'cytoplasm', filename), cytoplasm)
        cv2.imwrite(os.path.join(converted_path, 'labels', 'cervical_cell', filename), cell)

        filename_map[filename] = {
            'image': os.path.join(extracted_base, 'Dataset', 'Synthetic', 'testset.mat'),
            **{k: os.path.join(extracted_base, 'Dataset', 'Synthetic', 'testset_GT.mat') for k in ['cervical_cell', 'nucleus', 'cytoplasm']}
        } 
        split_map['test'].append(filename)     
        idx += 1
    

    save_mapping(converted_path, filename_map)
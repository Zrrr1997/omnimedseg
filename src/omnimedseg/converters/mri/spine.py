from omnimedseg.core.paths import data_root
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import SimpleITK as sitk

from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import matplotlib.pyplot as plt

def convert_spine():
    converted_path = os.path.join(data_root(), 'MRI', 'SPINE', 'converted')

    # Create label mapping
    label_classes = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'T10', 'T11', 'T12']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'SPINE', 'extracted')

    image_paths = []
    image_paths = [os.path.join(extracted_base, el) for el in os.listdir(os.path.join(extracted_base)) if '.dcm' in el]

    filename_map = {}


    def process_file(idx_file):
        idx, file = idx_file
        filename = f"{idx:06d}.nii.gz"

        dest_image_path = os.path.join(converted_path, 'images', filename)
        output_dir = file.replace('.dcm', '_nifti')
        output_base = os.path.basename(output_dir).split('_')[0]
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        temp_dict = {}
        subprocess.run([
            "dcm2niix", "-z", "y", "-o", output_dir, "-f", "%p_%s", file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        nii_files = [os.path.join(output_dir, el) for el in os.listdir(output_dir) if 'nii.gz' in el]
        source_path = output_dir.replace('_nifti', '')
        mask_file = os.path.join(source_path, f'L1.mha')
        label_shape = sitk.ReadImage(mask_file).GetSize()
        include_list = {'AKa2': '3_3a', 'AKa3': '4_4', 'AKa4': '5_6', 'AKs5': '5_5_1', 'AKs6': '4_4a', 'AKs7': '5_5', 'AKs8': '4_11', 'S01': '_401', 'S02': '_501', 'St1': '401b', 'F02':'401a', 'F03':'_2b', 'F04':'SAG_2', 'AKs3':'3_9', 'C002':'SAG_2a', 'DzZ_T1':'CLEAR_401', 'DzZ_T2':'CLEAR_301'}
        # aks3, dzz_t1, dzz_t2 - flip last - false
        for nii_file in nii_files:
            if 'DzZ' in nii_file:
                continue # DzZ files are not converted 
            if output_base in list(include_list.keys()):
                ending = include_list[output_base]
                if not nii_file.endswith(f'{ending}.nii.gz'):
                    continue
            nii_img = sitk.ReadImage(nii_file)
            flip_filter = sitk.FlipImageFilter()
            last_axis_flip = False if any(['AKs5' in nii_file, 'DzZ' in nii_file]) else True
            flip_filter.SetFlipAxes((False, True, last_axis_flip)) 
            nii_img_lps = flip_filter.Execute(nii_img)       


            label_img = sitk.ReadImage(mask_file)

            # Match geometry to the label
            nii_img_lps.SetOrigin(label_img.GetOrigin())
            nii_img_lps.SetSpacing(label_img.GetSpacing())
            nii_img_lps.SetDirection(label_img.GetDirection())

            sitk.WriteImage(nii_img_lps, nii_file)
            sitk.WriteImage(nii_img_lps, os.path.join(converted_path, 'images', filename))
            temp_dict['image'] = file
            for label_class in label_classes:
                mask_file = os.path.join(source_path, f'{label_class}.mha')
                if os.path.exists(mask_file):
                    label_img = sitk.ReadImage(mask_file)

                    binary_img = sitk.BinaryThreshold(
                        label_img,
                        lowerThreshold=1,
                        upperThreshold=255,  # anything > 0
                        insideValue=1,
                        outsideValue=0
                    )

                    sitk.WriteImage(binary_img, os.path.join(converted_path, 'labels', label_class, filename))
                    temp_dict[label_class] = mask_file


        

        return filename, temp_dict

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, (idx, file)) for idx, file in enumerate(image_paths)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing SPINE images"):
            filename, temp_dict = f.result()
            filename_map[filename] = temp_dict


    save_mapping(converted_path, filename_map)


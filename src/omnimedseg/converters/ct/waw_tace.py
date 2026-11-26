from omnimedseg.core.paths import data_root
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import SimpleITK as sitk
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

import nibabel as nib

def _process_waw_tace(args):
    idx, file, converted_path = args
    filename = f"{idx:06d}.nii.gz"
    data_dict = {}
    patient_id = os.path.basename(os.path.dirname(file))
    ref_img = nib.load(file)

    # Copy image
    dst_img = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, dst_img)
    data_dict['image'] = file
    data_dict['liver_tumor'] = []
    # Copy label
    label_file = file.replace(f'/{patient_id}/', f'/tumor_masks_wawtace_v1_08_05_2024/{patient_id}/').replace('scan.nii.gz', f'0_tumor_seg.nrrd')
    
    label_img = convert_nrrd_nii(label_file)
    label_data = label_img.get_fdata().astype(np.uint8)
    data_dict['liver_tumor'].append(label_file)
    label_id = 1

    if label_data.shape != nib.load(file).get_fdata().shape:
        resample_seg_to_image(label_file, file, label_file.replace('.nrrd', '.nii.gz'))
        label_img = nib.load(label_file.replace('.nrrd', '.nii.gz'))
        label_data = label_img.get_fdata().astype(np.uint8)

    while True:
        new_label_file = file.replace(f'/{patient_id}/', f'/tumor_masks_wawtace_v1_08_05_2024/{patient_id}/').replace('scan.nii.gz', f'{label_id}_tumor_seg.nrrd')
        if not os.path.exists(new_label_file):
            break
        else:
            # load and add the next mask
            new_label_img = convert_nrrd_nii(new_label_file)
            new_label_data = new_label_img.get_fdata().astype(np.uint8)
            if new_label_data.shape != label_data.shape:
                resample_seg_to_image(new_label_file, file, new_label_file.replace('.nrrd', '.nii.gz'))
                new_label_img = nib.load(new_label_file.replace('.nrrd', '.nii.gz'))
                new_label_data = new_label_img.get_fdata().astype(np.uint8)


            label_data += new_label_data 
            label_id += 1
            data_dict['liver_tumor'].append(new_label_file)

    final_img = nib.Nifti1Image(label_data, affine=ref_img.affine, header=ref_img.header)
    nib.save(final_img, os.path.join(converted_path, 'labels', 'liver_tumor', filename))


    return filename, data_dict

def convert_waw_tace(num_workers=8):
    converted_path = os.path.join(data_root(), 'CT', 'WAW_TACE', 'converted')
    label_classes = ['liver_tumor']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'CT', 'WAW_TACE', 'extracted')
    image_paths = []
    for i in range(0, 5):
        image_paths += [os.path.join(extracted_base, f, f'{f}_{i}_scan.nii.gz') for f in os.listdir(os.path.join(extracted_base)) if os.path.exists(os.path.join(extracted_base, 'tumor_masks_wawtace_v1_08_05_2024', f, f'{f}_{i}_0_tumor_seg.nrrd'))]

    filename_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_waw_tace, (i, file, converted_path)) for i, file in enumerate(image_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing WAW_TACE images"):
            filename, mapping = future.result()
            filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)


def resample_seg_to_image(seg_path, image_path, output_path):
    """
    Resample a segmentation NIfTI to exactly match the shape, spacing, origin, and direction of the image.

    Args:
        seg_path (str): Path to segmentation NIfTI.
        image_path (str): Path to reference image NIfTI.
        output_path (str): Path to save resampled segmentation.
    """
    # Read image and segmentation
    img = sitk.ReadImage(image_path)
    seg = sitk.ReadImage(seg_path)

    # Resample segmentation to image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # preserve labels
    resampler.SetDefaultPixelValue(0)                     # background label
    resampler.SetOutputSpacing(img.GetSpacing())
    resampler.SetSize(img.GetSize())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    seg_resampled = resampler.Execute(seg)
    # Make binary: 0 for background, 1 for any label
    seg_binary = sitk.BinaryThreshold(seg_resampled, lowerThreshold=1, upperThreshold=10, insideValue=1, outsideValue=0)

    # Save output
    sitk.WriteImage(seg_binary, output_path)
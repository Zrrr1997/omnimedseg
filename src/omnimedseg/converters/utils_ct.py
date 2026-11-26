import os
import threading
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import nrrd
import nibabel as nib
import SimpleITK as sitk
import pydicom
from nibabel.processing import resample_from_to

from omnimedseg.utils.dataset_utils import *
from omnimedseg.utils.postprocess_download import *
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from multiprocessing import cpu_count

sitk.ProcessObject.SetGlobalWarningDisplay(False)
logging.getLogger('pydicom').setLevel(logging.ERROR)





def print_patient_id(dicom_dir):
    for filename in os.listdir(dicom_dir):
        if filename.lower().endswith(".dcm"):
            filepath = os.path.join(dicom_dir, filename)
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(ds, "PatientID"):
                    return ds.PatientID  # Return on first found
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return None


def convert_mhd_raw_to_nifti(mhd_path, output_path=None, label_classes=None, label_indices=None):
    # Read the image from .mhd/.raw pair
    image = sitk.ReadImage(mhd_path)

    # Define output path
    if output_path is None:
        output_path = os.path.splitext(mhd_path)[0] + ".nii.gz"
    
    if label_classes is None:
        # Write the image in NIfTI format
        sitk.WriteImage(image, output_path)
    else:
        for i, label_id in enumerate(label_indices):
            label_name = label_classes[i]
            label_image = sitk.Equal(image, float(label_id), 0, 1)
            label_image = sitk.Cast(label_image, sitk.sitkUInt8)
            out_file = output_path.replace('temp', label_name)
            sitk.WriteImage(label_image, out_file)
        if 'lungs' in label_classes:
            lungs_image = sitk.Or(sitk.Equal(image, float(label_indices[label_classes.index('left_lung')]), 0, 1),
                                  sitk.Equal(image, float(label_indices[label_classes.index('right_lung')]), 0, 1))
            lungs_image = sitk.Cast(lungs_image, sitk.sitkUInt8)
            out_file = output_path.replace('temp', 'lungs')
            sitk.WriteImage(lungs_image, out_file)

    return output_path



def convert_nrrd_nii(nrrd_path, flip=False):
    # Load NRRD file
    data, header = nrrd.read(nrrd_path)

    # Extract affine if available (otherwise use identity)
    # NRRD doesn't always store affine explicitly, so we try to construct it
    if 'space directions' in header and 'space origin' in header:
        spacing = np.array(header['space directions'])
        origin = np.array(header['space origin'])
        affine = np.eye(4)
        affine[:3, :3] = spacing
        affine[:3, 3] = origin

    else:
        affine = np.eye(4)  # default to identity if metadata is missing
    # Create a NIfTI image
    nii = nib.Nifti1Image(data, affine)

    return nii

def unzip_single_tcia_folder(file, extracted_base):
    zip_file = os.path.join(file, f'{os.path.basename(file)}.zip')
    dicom_dir = os.path.join(file, os.path.basename(file)).replace('compressed', 'extracted')

    if not is_unzipped(dicom_dir, zip_file):
        unzip_file(zip_file, dicom_dir, verbose=False)
    return zip_file  # Optional return


def unzip_tcia(extracted_base, compressed_base):
    os.makedirs(extracted_base, exist_ok=True)

    zip_paths = [os.path.join(compressed_base, f) for f in os.listdir(compressed_base)]

    # Parallel unzip with progrsess bar
    with ProcessPoolExecutor() as executor:
        worker = partial(unzip_single_tcia_folder, extracted_base=extracted_base)
        for _ in tqdm(executor.map(worker, zip_paths), total=len(zip_paths), desc="📦 Unzipping TCIA folders"):
            pass

def is_rtstruct(dicom_dir):
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if getattr(ds.SOPClassUID, "name", "") == "RT Structure Set Storage":
                return True
        except Exception:
            # Skip non-DICOM or corrupted files
            continue
    return False


def is_dicom_seg(dicom_dir):
    try:
        files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        for f in files:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if getattr(ds, "Modality", None) == "SEG":
                return True
    except Exception as e:
        print(f"⚠️ Could not check for DICOM-SEG in {dicom_dir}: {e}")
    return False

def pediatric_label_map(nifti_seg):
    if 'Adrenal_Left' in nifti_seg or 'lt adrenal' in nifti_seg:
        return 'left_adrenal_gland'
    elif 'Adrenal_Right' in nifti_seg or 'rt adrenal' in nifti_seg:
        return 'right_adrenal_gland'
    elif 'Breast_Left' in nifti_seg or 'breast left' in nifti_seg:
        return 'left_breast'
    elif 'Breast_Right' in nifti_seg or 'breast right' in nifti_seg:
        return 'right_breast'
    elif 'Femoral_Head_Lef' in nifti_seg:
        return 'left_femoral_head'
    elif 'Femoral_Head_Rig' in nifti_seg:
        return 'right_femoral_head'
    elif 'Gall' in nifti_seg:
        return 'gallbladder'
    elif 'Kidney_Left' in nifti_seg or 'kidney left' in nifti_seg:
        return 'left_kidney'
    elif 'Kidney_Right' in nifti_seg or 'kidney right' in nifti_seg:
        return 'right_kidney'
    elif 'Large Intestine' in nifti_seg:
        return 'large_intestine'
    elif 'Lung_L' in nifti_seg:
        return 'left_lung'
    elif 'Lung_R' in nifti_seg:
        return 'right_lung'
    elif 'Small Intestine' in nifti_seg:
        return 'small_intestine'
    elif 'Spinal Canal' in nifti_seg:
        return 'spinal_canal'
    elif 'Utero' in nifti_seg:
        return 'utero_cervix'
    else:
        return os.path.basename(nifti_seg).split('.nii')[0].lower()
def lctsc_label_map(nifti_seg):
    if 'Lung_L' in nifti_seg:
        return 'left_lung'
    elif 'Lung_R' in nifti_seg:
        return 'right_lung'
    elif 'SpinalCord' in nifti_seg:
        return 'spinal_cord'
    else:
        return os.path.basename(nifti_seg).split('.nii.gz')[0].replace('Struct_', '').lower()
def pancreatic_label_map(nifti_seg):
    if 'Bowel_sm' in nifti_seg:
        return 'small_bowel'
    elif 'LUNG_R' in nifti_seg:
        return 'right_lung'
    elif 'LUNG_L' in nifti_seg:
        return 'left_lung'
    elif 'Stomach' in nifti_seg:
        return 'stomach_duodenum'


def process_dicom_case(file, name, extracted_base, converted_path):
    dicom_dir = os.path.join(file, os.path.basename(file)) if 'DICOM_anon' not in file else file

    try:
        patient_id = print_patient_id(dicom_dir=dicom_dir)
    except Exception as e:
        #print(f"⚠️ Failed to get patient ID for {dicom_dir}: {e}")
        return

    output_dir = os.path.join(converted_path, 'all_niftis', patient_id)
    os.makedirs(output_dir, exist_ok=True)
    curr_output_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
    output_files = []

    if is_rtstruct(dicom_dir) or is_dicom_seg(dicom_dir):
        rtstruct_file = os.listdir(dicom_dir)[0]
        shutil.copy(os.path.join(dicom_dir, rtstruct_file), os.path.join(output_dir, rtstruct_file))
        with open(os.path.join(output_dir, 'log.txt'), 'a+') as f:
            f.write(f"{dicom_dir} -> {os.path.join(output_dir, rtstruct_file)}\n")
        return

    if name == 'HCC_TACE_SEG':
        return

    #  Not a SEG, proceed with dcm2niix
    try:
        subprocess.run([
            "dcm2niix", "-z", "y", "-o", output_dir, "-f", "%p_%s", dicom_dir
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        new_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith('.nii.gz') and f not in curr_output_files
        ]
        output_files.extend(new_files)

    except subprocess.CalledProcessError:
        try:

            # Fallback: CT or MRI
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
            if not series_ids:
                raise RuntimeError("No DICOM series found.")
            series_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(series_files)
            image = reader.Execute()
            output_file = os.path.join(output_dir, f"{patient_id}.nii.gz")
            sitk.WriteImage(image, output_file)
            output_files.append(output_file)

        except Exception as e:
            print(f"❌ Failed to convert {dicom_dir} with both dcm2niix and SimpleITK: {e}")
            return

    #  Log results
    try:
        with open(os.path.join(output_dir, 'log.txt'), 'a+') as f:
            for out in output_files:
                f.write(f"{dicom_dir} -> {out}\n")
    except Exception as e:
        print(f"⚠️ Failed to log {dicom_dir}: {e}")


def convert_extracted_dicoms(name, extracted_base, converted_path):
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base)]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_dicom_case, file, name, extracted_base, converted_path): file
            for file in image_paths
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"🖼️  Converting {name} images"):
            result = future.result()




def process_nifti_case(file, idx, name, converted_path, label_substring, label_name):
    filename = f"{idx:06d}.nii.gz"
    temp_dict = {}

    nifti_files = [os.path.join(file, el) for el in os.listdir(file) if '.nii.gz' in el]
    nifti_segs = [el for el in nifti_files if label_substring in el]
    nifti_files = [el for el in nifti_files if 'Segmentation' not in el and 'Struct' not in el and 'dicom_seg' not in el]
    if name == 'PEDIATRIC_CT_SEG':
        if not os.path.exists(os.path.join(file, 'rtstruct_outputs')):
            return None
        nifti_segs = [os.path.join(file, 'rtstruct_outputs', el) for el in os.listdir(os.path.join(file, 'rtstruct_outputs')) if '.nii' in el]

    if len(nifti_segs) == 0:
        return None  # Skip

    if name in ['ADRENAL_ACC_KIT67']:
        nifti_segs = [nifti_segs[0]]

    for nifti_file in nifti_files:
        image = nib.load(nifti_file)
        for nifti_seg in nifti_segs:
            seg_shape = nib.load(nifti_seg).get_fdata().shape

            if image.shape[:3] != seg_shape[:3]:
                continue

            image_target = os.path.join(converted_path, 'images', filename)
            if not os.path.exists(image_target):
                shutil.copy(nifti_file, image_target)

            if name == 'NSCLC_RADIOMICS':
                label_name_local = os.path.basename(nifti_seg).split('truct_')[-1].split('.nii')[0]
                label_map = {'Esophagus': 'esophagus', 'Heart': 'heart', 'Lung-Left': 'left_lung', 'Lung-Right': 'right_lung', 'Lungs-Total': 'lungs', 'Spinal-Cord': 'spinal_cord'}
                label_name_local = 'lung_tumor' if 'gtv' in nifti_seg.lower() else label_map.get(label_name_local, label_name_local)
            elif name == 'HCC_TACE_SEG':
                ind = int(os.path.basename(nifti_seg).split('dicom_seg_')[1][0])
                label_name_local = ['liver', 'liver_tumor', 'liver_vessel', 'aorta'][ind-1]
            elif name == 'COLORECTAL_LIVER_METASTASES':
                ind = int(os.path.basename(nifti_seg).split('dicom_seg_')[1][0])
                label_name_local = ['liver', 'liver_remnant', 'liver_vein', 'portal_vein', 'liver_tumor'][ind-1]
            elif name == 'QIN_PROSTATE':
                ind = int(os.path.basename(nifti_seg).split('dicom_seg_')[1][0])
                label_names = [
                    "prostate_normal_tissue",
                    "prostate_peripheral_zone",
                    "prostate_tumor",
                    "prostate",
                ]
                label_name_local = label_names[(ind) % len(label_names)]
            elif name == 'CT_LYMPH_NODES' or name == 'LNQ_2023':
                label_name_local = 'lymph_node'
            elif name == 'SPINEMETS':
                label_names = []
                for i in range(1, 13):
                    label_names.append(f'T{i}')
                for i in range(13, 18):
                    label_names.append(f'L{i - 12}')
                ind = int(os.path.basename(nifti_seg).split('dicom_seg_')[1].split('.nii.gz')[0])
                label_name_local = label_names[ind]
            elif name == 'PROSTATE_EDGE_CASES':
                label_name_local = None
                if 'Bladder' in nifti_seg:
                    label_name_local = 'bladder'
                elif 'Femur_Head_R' in nifti_seg:
                    label_name_local = 'right_femoral_head'
                elif 'Femur_Head_L' in nifti_seg:
                    label_name_local = 'left_femoral_head'
                elif 'Prostate' in os.path.basename(nifti_seg):
                    label_name_local = 'prostate'
                elif 'Rectum' in nifti_seg:
                    label_name_local = 'rectum'
                if label_name_local is None:
                    continue
            elif name == 'PEDIATRIC_CT_SEG':
                label_name_local = pediatric_label_map(nifti_seg)
            elif name == 'LCTSC':
                label_name_local = lctsc_label_map(nifti_seg)
            elif name == 'PANCREATIC_CT_CBCT_SEG':
                label_name_local = pancreatic_label_map(nifti_seg)
            else:
                label_name_local = label_name

            target_path = os.path.join(converted_path, 'labels', label_name_local, filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if nifti_seg.endswith(".nii") and not nifti_seg.endswith(".nii.gz"):
                img = sitk.ReadImage(nifti_seg)
                compressed_path = target_path if target_path.endswith(".nii.gz") else target_path + ".gz"
                sitk.WriteImage(img, compressed_path)
            else:
                shutil.copy(nifti_seg, target_path)

            try:
                with open(os.path.join(file, 'log.txt'), 'r') as f:
                    lines = f.readlines()
                    seg_source = [el.rstrip().split('->')[0] for el in lines if nifti_seg in el]
                    if len(seg_source) == 0:
                        seg_source = [el.rstrip().split('->')[0] for el in lines if '.dcm' in el] # DICOM_SEG
                    image_source = [el.rstrip().split('->')[0] for el in lines if nifti_file in el] 
                    temp_dict['image'] = image_source
                    temp_dict[label_name_local] = seg_source
            except FileNotFoundError:
                pass

    return (filename, temp_dict)


def match_and_save_niftis(name, converted_path, label_substring, label_name):

    filename_map = {}
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el) for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {
            executor.submit(process_nifti_case, file, idx, name, converted_path, label_substring, label_name): idx
            for idx, file in enumerate(nifti_paths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"🖼️  Matching {name} images"):
            result = future.result()
            if result:
                filename, mapping = result
                filename_map[filename] = mapping

    save_mapping(converted_path, filename_map)

def process_rtstruct(file, label_substring):
    try:
        nifti_file = next(
            os.path.join(file, el)
            for el in os.listdir(file)
            if el.endswith('.nii.gz') and label_substring not in el and 'Struct' not in el
        )
        rtstruct_file = next(
            os.path.join(file, el)
            for el in os.listdir(file)
            if el.endswith('.dcm')
        )

        with open(os.path.join(file, 'log.txt'), 'r') as f:
            lines = f.readlines()
            image_source = next(
                el.rstrip().split('->')[0] for el in lines if nifti_file in el
            ).rstrip()

        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(image_source)

        if not series_IDs:
            return None  # Skip if no series

        ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
        if getattr(ds.SOPClassUID, "name", "") == "RT Structure Set Storage":

            plastimatch_exe = shutil.which("plastimatch")
            if plastimatch_exe and 'PEDIATRIC_CT_SEG' in file:
                try:
                    # Ensure output directory exists
                    output_dir = os.path.join(file, 'rtstruct_outputs')
                    os.makedirs(output_dir, exist_ok=True)

                    cmd = [
                        plastimatch_exe, "convert",
                        "--input", rtstruct_file,
                        "--referenced-ct", image_source,
                        "--output-prefix", output_dir,
                        "--prefix-format", "nii"
                    ]
                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True
                    )
                    return
                except subprocess.CalledProcessError as e:
                    shutil.rmtree(file) 
                    return   
            else:
                if 'PEDIATRIC_CT_SEG' in file:
                    print("Please install plastimatch for converting PEDIATRIC_CT_SEG!")
                    return
                elif 'PANCREATIC_CT_CBCT_SEG' in file:
                    rtstruct_files = [
                        os.path.join(file, el)
                        for el in os.listdir(file)
                        if el.endswith('.dcm')
                    ]
                    for i, rtstruct_file in enumerate(rtstruct_files):
                        os.makedirs(os.path.join(file, f'{i}'), exist_ok=True)
                        rt_path = os.path.join(file, f'{i}')
                        convert_rtstruct(image_source, rtstruct_file, prefix="Struct_", output_dir=rt_path)
                        stay = False
                        for nii in os.listdir(rt_path):
                            if 'planCT' in nii:
                                stay = True
                        if stay:
                            for nii in os.listdir(rt_path):
                                if 'ROI' not in nii:
                                    shutil.move(os.path.join(rt_path, nii), os.path.join(os.path.dirname(rt_path), nii))
                        else:
                            os.remove(rtstruct_file)
                        shutil.rmtree(rt_path)
                    return 
                elif 'CC_TUMOR_HETEROGENEITY' in file or 'CROSSMODA' in file:
                    if 'CROSSMODA' in file:
                        dataset = 'CROSSMODA'
                    else:
                        dataset = 'CC_TUMOR_HETEROGENEITY'
                    rtstruct_files = [
                        os.path.join(file, el)
                        for el in os.listdir(file)
                        if el.endswith('.dcm')
                    ]
                    valid_series_uid = []
                    study_ids = []
                    log_file = os.path.join(file, 'log.txt')

                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        nii_files = [os.path.join(file, nii_file) for nii_file in os.listdir(file) if '.nii.gz' in nii_file]

                    for i, rtstruct_file in enumerate(rtstruct_files):
                        ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
                        study_ids.append(ds.StudyInstanceUID)

                        frame_uid = ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID

                        # Referenced Series (MRI series)
                        ref_series = ds.ReferencedFrameOfReferenceSequence[0] \
                            .RTReferencedStudySequence[0] \
                            .RTReferencedSeriesSequence[0]

                        series_uid = ref_series.SeriesInstanceUID
                        valid_series_uid.append(series_uid)

                        image_source = os.path.join('datasets', 'MRI', dataset, 'extracted', series_uid, series_uid)
                        os.makedirs(os.path.join(file, f'{i}'), exist_ok=True)
                        rt_path = os.path.join(file, f'{i}')
                        if dataset == 'CROSSMODA':
                            prefix = 'Struct_' + ds.StructureSetLabel
                        else:
                            prefix = 'Struct_'
                        convert_rtstruct(image_source, rtstruct_file, prefix=prefix, output_dir=rt_path)
                        with open(os.path.join(file, 'log.txt'), 'a+') as f:
                            dicom_src = [el.split('->')[0].rstrip() for el in lines if rtstruct_file in el][0]
                            out_file = [os.path.join(rt_path, el) for el in os.listdir(rt_path) if 'Struct_' in el][0]
                            f.write(f"{dicom_src} -> {out_file}\n")

                    for nii_file in nii_files:
                        stay = False
                        sources = [el for el in lines if nii_file in el]

                        for valid_uid in valid_series_uid:
                            for source in sources:
                                if valid_uid in source:
                                    stay = True
                        if not stay:
                            if os.path.exists(nii_file):
                                os.remove(nii_file)
                                os.remove(nii_file.replace('.nii.gz', '.json'))
                        else:
                            # find the first DICOM file in the folder
                            for fname in os.listdir(sources[0].split('->')[0].rstrip()):
                                fpath = os.path.join(sources[0].split('->')[0].rstrip(), fname)
                                if os.path.isfile(fpath):
                                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                                    for i in range(len(study_ids)):
                                        if ds.StudyInstanceUID == study_ids[i]:
                                            shutil.copy(nii_file, os.path.join(os.path.join(os.path.dirname(rt_path), f'{i}'), os.path.basename(nii_file)))
                                break



                        

                else:
                    convert_rtstruct(image_source, rtstruct_file, prefix="Struct_", output_dir=file)
                    return


        else:
            return None
    except Exception as e:
        print(f"❌ Failed processing {file}: {e}")
        return None

def convert_rtstructs(name, converted_path, label_substring):
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el)
                   for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    worker = partial(process_rtstruct, label_substring=label_substring)

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker, nifti_paths), total=len(nifti_paths),
                  desc=f"🖼️  Converting {name} RTSTRUCT files"))

def reorder_files_and_json_inplace(image_dir, json_path):
    # Load JSON
    if json_path is not None:
        with open(json_path, "r") as f:
            data = json.load(f)

    # Collect and sort files
    files = [f for f in os.listdir(image_dir) if f.endswith(".nii.gz")]
    files.sort()  # zero-padded names sort correctly

    mapping = {}  # old -> new filename mapping

    for new_idx, fname in enumerate(files):
        new_name = f"{new_idx:06d}.nii.gz"
        mapping[fname] = new_name

        src = os.path.join(image_dir, fname)
        dst = os.path.join(image_dir, new_name)

        if src != dst:
            # Use a temporary name if renaming could collide
            temp_dst = dst + ".tmp"
            os.rename(src, temp_dst)

    # Rename temp files to final names
    for temp_file in os.listdir(image_dir):
        if temp_file.endswith(".tmp"):
            final_name = temp_file[:-4]  # remove .tmp
            os.rename(os.path.join(image_dir, temp_file),
                      os.path.join(image_dir, final_name))

    if json_path is not None:
        # Update JSON keys
        new_data = {}
        for old_name, new_name in mapping.items():
            if old_name in data:
                new_data[new_name] = data[old_name]
        # Overwrite JSON
        with open(json_path, "w") as f:
            json.dump(new_data, f, indent=2)


def process_file_msd(file, idx, converted_path, filename_map, lock, targets, images_dir='imagesTr', labels_dir='labelsTr'):
    filename = f"{idx:06d}.nii.gz"
    image_dest = os.path.join(converted_path, 'images', filename)
    shutil.copy(file, image_dest)

    if type(targets) == str:

        label_dest = os.path.join(converted_path, 'labels', targets, filename)

        shutil.copy(file.replace(images_dir, labels_dir), label_dest)
        with lock:
            filename_map[filename] = {
                'image': file,
                targets: file.replace(images_dir, labels_dir)
            }
    else:
        label = sitk.ReadImage(file.replace(images_dir, labels_dir))
        for i, target in enumerate(targets):
            label_ = sitk.Equal(label, float(i + 1), 0, 1)
            label_ = sitk.Cast(label_, sitk.sitkUInt8)
            sitk.WriteImage(label_, os.path.join(converted_path, 'labels', target, filename))
        filename_map[filename] = {
            'image': file,
            **{k: file.replace(images_dir, labels_dir) for k in targets}
        }


def convert_msd(name, target, subdir, images_dir='imagesTr', labels_dir='labelsTr', start=0, modality='CT'):
    converted_path = os.path.join('datasets', modality, name, 'converted')
    label_classes = [target] if type(target) == str else target
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join('datasets', modality, name, 'extracted', subdir, images_dir)
    image_paths = [os.path.join(extracted_base, f) for f in os.listdir(extracted_base) if '._' not in f and '.DS' not in f]

    filename_map = {}
    if os.path.exists(os.path.join(converted_path, 'filename_map.json')):
        with open(os.path.join(converted_path, 'filename_map.json'), 'r') as f:
            filename_map = json.load(f)
    lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        list(tqdm(
            executor.map(lambda args: process_file_msd(*args, converted_path, filename_map, lock, target, images_dir, labels_dir),
                         [(file, idx) for idx, file in enumerate(image_paths, start=start)]),
            total=len(image_paths),
            desc=f"🖼️  Processing {name} images"
        ))

    save_mapping(converted_path, filename_map)



def align_label_to_image(image_path, label_path, aligned_label_path):
    """
    Align label to image space 
    """
    # Load image
    img = nib.load(image_path)

    # Check for 4D image
    if len(img.shape) == 4:
        # Extract first volume for 3D processing
        img_3d_data = img.get_fdata()[..., 0]
        img_affine = img.affine[:4, :4]
        img_header = img.header
        img = nib.Nifti1Image(img_3d_data, img_affine, img_header)

    # Load label and IMMEDIATELY force it to use image's geometry
    label = nib.load(label_path)
    if len(label.shape) == 4:
        label_data = label_data[..., 0]
    else:
        label_data = label.get_fdata()
    
    if label_data.shape[0] == img.shape[2] and label_data.shape[2] != img.shape[2]:
        label_data = np.transpose(label_data, (1, 2, 0))

    fixed_label = nib.Nifti1Image(label_data, img.affine, img.header)


    
    # Check if resampling is actually needed
    if fixed_label.shape == img.shape and np.allclose(fixed_label.affine, img.affine):
        # Already aligned! Just save the fixed label
        nib.save(fixed_label, aligned_label_path)
    else:
        # Resample needed
        aligned_label = resample_from_to(fixed_label, img, order=0, mode='constant', cval=0)
        
        # Ensure header matches image exactly
        aligned_label = nib.Nifti1Image(aligned_label.get_fdata(), img.affine, img.header)
        nib.save(aligned_label, aligned_label_path)
    
    
    return image_path, aligned_label_path


def get_image_label_pairs(converted_path):
    """Get all image-label pairs from converted directory"""
    image_dir = os.path.join(converted_path, 'images')
    label_dir = os.path.join(converted_path, 'labels')

    pairs = []
    for fname in os.listdir(image_dir):
        if fname.endswith('.nii.gz'):
            image_path = os.path.join(image_dir, fname)
            for label_subdir in os.listdir(label_dir): 
                label_subdir_path = os.path.join(label_dir, label_subdir)
                if os.path.isdir(label_subdir_path):
                    potential_label_path = os.path.join(label_subdir_path, fname)
                    if os.path.exists(potential_label_path):
                        pairs.append((image_path, potential_label_path))
    return pairs

def process_single_pair(args):
    """Process a single image-label pair (for parallel processing)"""
    image_path, label_path = args
    try:
        # aligned_label_path is the same as label_path (overwrite)
        align_label_to_image(image_path, label_path, label_path)
        return True, image_path, None
    except Exception as e:
        return False, image_path, str(e)

def align_all_pairs(converted_path, max_workers=None, show_progress=True):
    """
    Process all image-label pairs in parallel.
    
    Args:
        converted_path: Path to converted dataset directory
        max_workers: Number of parallel workers (default: CPU count)
        show_progress: Show progress bar
    
    Returns:
        tuple: (successful_count, failed_count, failed_pairs)
    """
    # Get all pairs
    pairs = get_image_label_pairs(converted_path)
    
    if not pairs:
        print(f"⚠️ No image-label pairs found in {converted_path}")
        return 0, 0, []
    
    print(f"📊 Found {len(pairs)} image-label pairs to process")
    
    successful = 0
    failed = 0
    failed_pairs = []
    
    if show_progress:
        print(f"🚀 Processing with {max_workers or 'all available'} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_pair, pair): pair for pair in pairs}
        
        # Process with progress bar
        if show_progress:
            with tqdm(total=len(pairs), desc="Aligning labels", unit="pair") as pbar:
                for future in as_completed(futures):
                    success, image_path, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        failed_pairs.append((image_path, error))
                    
                    pbar.set_postfix({'success': successful, 'failed': failed})
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                success, image_path, error = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    failed_pairs.append((image_path, error))
    
    # Print summary
    if show_progress:        
        if failed_pairs:
            print(f"\n⚠️ Failed pairs:")
            for image_path, error in failed_pairs[:5]:  # Show first 5 failures
                print(f"   - {os.path.basename(image_path)}: {error}")
            if len(failed_pairs) > 5:
                print(f"   ... and {len(failed_pairs) - 5} more")
    
    return successful, failed, failed_pairs







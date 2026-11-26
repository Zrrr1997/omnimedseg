from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *

def convert_adrenal_acc_kit67():
    converted_path = os.path.join(data_root(), 'CT', 'ADRENAL_ACC_KIT67', 'converted')
    label_classes = ['adrenocortical_carcinoma']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join(data_root(), 'CT', 'ADRENAL_ACC_KIT67', 'compressed')
    extracted_base = os.path.join(data_root(), 'CT', 'ADRENAL_ACC_KIT67', 'extracted')
    unzip_tcia(extracted_base, compressed_base)

    convert_extracted_dicoms('ADRENAL_ACC_KIT67', extracted_base, converted_path)
    convert_dicom_segs('ADRENAL_ACC_KIT67', converted_path)

    match_and_save_niftis('ADRENAL_ACC_KIT67', converted_path, 'dicom_seg', 'adrenocortical_carcinoma')
    reorder_files_and_json_inplace(os.path.join(converted_path, 'images'), os.path.join(converted_path, 'filename_map.json'))
    reorder_files_and_json_inplace(os.path.join(converted_path, 'labels', 'adrenocortical_carcinoma'), None)

def convert_dicom_segs(name, converted_path):
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el)
                   for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    worker = partial(process_dicom_seg)

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker, nifti_paths), total=len(nifti_paths),
                  desc=f"🖼️  Converting {name} DICOM_SEG files"))


def process_dicom_seg(path):
    try:
        if len([el for el in os.listdir(path) if el.endswith('.dcm')]) == 0:
            shutil.rmtree(path)
            return None
        # Get the DICOM SEG file
        dicom_seg_file = next(
            os.path.join(path, el)
            for el in os.listdir(path)
            if el.endswith('.dcm')
        )

        cmd = [
            "segimage2itkimage",
            "--inputDICOM", dicom_seg_file,
            "--outputDirectory", os.path.dirname(dicom_seg_file),
            "-t", "nii"
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(os.path.join(path, '1.nii.gz')):
            shutil.move(os.path.join(path, '1.nii.gz'), os.path.join(path, 'dicom_seg_0.nii.gz'))
        log_file = os.path.join(path, 'log.txt')
        with open(log_file, 'r') as f:
            dicom_source = [el.split('->')[0].rstrip() for el in f.readlines() if dicom_seg_file in el][0]
        with open(log_file, 'a+') as f:
            f.write(f"{dicom_source} -> {os.path.join(path, 'dicom_seg_0.nii.gz')}\n")

    except Exception as e:
        print(f"❌ Failed to process DICOM SEG {path}: {e}")
        return None

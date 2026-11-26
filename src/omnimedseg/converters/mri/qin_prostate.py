from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor


def convert_qin_prostate():
    converted_path = os.path.join(data_root(), 'MRI', 'QIN_PROSTATE', 'converted')
    label_classes = ['prostate_peripheral_zone', 'prostate', 'prostate_tumor', 'prostate_normal_tissue']
    ensure_directories(converted_path, label_classes)

    extracted_base = os.path.join(data_root(), 'MRI', 'QIN_PROSTATE', 'extracted')
    compressed_base = os.path.join(data_root(), 'MRI', 'QIN_PROSTATE', 'compressed')
    unzip_tcia(extracted_base, compressed_base)
    convert_extracted_dicoms('QIN_PROSTATE', extracted_base, converted_path)
    convert_dicom_segs('QIN_PROSTATE', converted_path)
    match_and_save_niftis_qin_prostate(converted_path)

def convert_dicom_segs_parallel(name, converted_path):
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el)
                   for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    worker = partial(process_dicom_seg)

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker, nifti_paths), total=len(nifti_paths),
                  desc=f"Converting {name} DICOM_SEG files"))
def convert_dicom_segs(name, converted_path):
    nifti_paths = [os.path.join(converted_path, 'all_niftis', el)
                   for el in os.listdir(os.path.join(converted_path, 'all_niftis'))]

    for path in tqdm(nifti_paths, total=len(nifti_paths),
                     desc=f"Converting {name} DICOM_SEG files"):
        process_dicom_seg(path)


def process_dicom_seg(path):
    try:
        if len([el for el in os.listdir(path) if el.endswith('.dcm')]) == 0:
            shutil.rmtree(path)
            return None
        # Get the DICOM SEG file
        dicom_seg_files = [os.path.join(path, el) for el in os.listdir(path) if el.endswith('.dcm')]

        for dicom_seg_file in dicom_seg_files:
            cmd = [
                "segimage2itkimage",
                "--inputDICOM", dicom_seg_file,
                "--outputDirectory", os.path.dirname(dicom_seg_file),
                "-t", "nii"
            ]

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            all_dicom_seg_niftis = [int(el.split('.nii')[0].split('_')[-1]) for el in os.listdir(path) if 'dicom_seg' in el]
            if len(all_dicom_seg_niftis) > 0:
                max_index = max(all_dicom_seg_niftis)
                max_index = ((max_index // 4) + 1) * 4
            else:
                max_index = 0


            for i in range(1, 5):
                if os.path.exists(os.path.join(path, f'{i}.nii.gz')):
                    shutil.move(os.path.join(path, f'{i}.nii.gz'), os.path.join(path, f'dicom_seg_{max_index + i - 1}.nii.gz'))
                log_file = os.path.join(path, 'log.txt')
                with open(log_file, 'r') as f:
                    dicom_source = [el.split('->')[0].rstrip() for el in f.readlines() if dicom_seg_file in el][0]
                with open(log_file, 'a+') as f:
                    f.write(f"{dicom_source} -> {os.path.join(path, f'dicom_seg_{max_index + i - 1}.nii.gz')}\n")
            

    except Exception as e:
        print(f"Failed to process DICOM SEG {path}: {e}")
        return None



def match_and_save_niftis_qin_prostate(converted_path):

    filename_map = {}
    nifti_paths = [
        os.path.join(converted_path, "all_niftis", el)
        for el in os.listdir(os.path.join(converted_path, "all_niftis"))
    ]

    global_idx = 0
    seen_files = []
    mapped_files = {}
    # Process files sequentially instead of in parallel
    for file in tqdm(nifti_paths, desc="Matching QIN_PROSTATE"):
        results = process_nifti_case_qin_prostate(file)

        if not results:
            continue

        for res in results:
            nifti_file, nifti_seg, label_name_local, mapping = res
            
            if nifti_file not in seen_files:
                seen_files.append(nifti_file)
                filename = f"{global_idx:06d}.nii.gz"
                image_target = os.path.join(converted_path, "images", filename)
                shutil.copy(nifti_file, image_target)
                global_idx += 1
                mapped_files[nifti_file] = filename
                filename_map[filename] = mapping

            else:
                filename = mapped_files[nifti_file]
                for k in mapping:
                    if k not in filename_map[filename]:
                        filename_map[filename][k] = mapping[k]


            target_path = os.path.join(
                converted_path,
                "labels",
                label_name_local,
                filename
            )

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(nifti_seg, target_path)


    save_mapping(converted_path, filename_map)


def match_images_to_segs(nifti_files, nifti_segs):
    import nibabel as nib
    import numpy as np

    # preload images (avoid reloading many times)
    images = [(img_path, nib.load(img_path)) for img_path in nifti_files]

    used_images = set()
    pairs = []

    for seg_path in nifti_segs:
        seg = nib.load(seg_path)

        best_img = None
        best_score = np.inf

        for img_path, image in images:

            if img_path in used_images:
                continue

            # shape check (fast skip)
            if image.shape[:3] != seg.shape[:3]:
                continue

            # --- same scoring as before ---
            img_zooms = np.array(image.header.get_zooms()[:3])
            seg_zooms = np.array(seg.header.get_zooms()[:3])
            zoom_diff = np.sum(np.abs(img_zooms - seg_zooms))

            affine_diff = np.sum(np.abs(image.affine - seg.affine))

            origin_diff = np.linalg.norm(
                image.affine[:3, 3] - seg.affine[:3, 3]
            )

            img_ornt = nib.orientations.aff2axcodes(image.affine)
            seg_ornt = nib.orientations.aff2axcodes(seg.affine)
            orientation_penalty = 0 if img_ornt == seg_ornt else 1000

            score = affine_diff + 10 * zoom_diff + 5 * origin_diff + orientation_penalty

            if score < best_score:
                best_score = score
                best_img = img_path

        if best_img is not None:
            used_images.add(best_img)
            pairs.append((best_img, seg_path))
        else:
            print(f"No match found for seg: {seg_path}")

    return pairs

def process_nifti_case_qin_prostate(file):

    results = []

    nifti_files = [
        os.path.join(file, el)
        for el in os.listdir(file)
        if el.endswith(".nii.gz")
    ]

    nifti_segs = [el for el in nifti_files if "dicom_seg" in el]

    nifti_files = [
        el for el in nifti_files
        if "Segmentation" not in el
        and "Struct" not in el
        and "dicom_seg" not in el
    ]

    if len(nifti_segs) == 0:
        return None

    label_names = [
        "prostate_normal_tissue",
        "prostate_peripheral_zone",
        "prostate_tumor",
        "prostate",
    ]


    paired_nifti_segs = [el for el in nifti_segs if int(el.split("dicom_seg_")[1].split(".nii.gz")[0]) % 4 == 0]


    for i, nifti_seg in enumerate(paired_nifti_segs):
        shutil.rmtree(os.path.join(file, str(i)), ignore_errors=True)
        os.makedirs(os.path.join(file, str(i)), exist_ok=True)
        seg = nib.load(nifti_seg)
        shutil.copy(nifti_seg, os.path.join(file, str(i), os.path.basename(nifti_seg)))
        for j in range(1, 4):
            if os.path.exists(nifti_seg.replace(f"dicom_seg_{int(nifti_seg.split('dicom_seg_')[1].split('.nii.gz')[0])}", f"dicom_seg_{int(nifti_seg.split('dicom_seg_')[1].split('.nii.gz')[0]) + j}")):
                shutil.copy(nifti_seg.replace(f"dicom_seg_{int(nifti_seg.split('dicom_seg_')[1].split('.nii.gz')[0])}", f"dicom_seg_{int(nifti_seg.split('dicom_seg_')[1].split('.nii.gz')[0]) + j}"), os.path.join(file, str(i), os.path.basename(nifti_seg.replace(f"dicom_seg_{int(nifti_seg.split('dicom_seg_')[1].split('.nii.gz')[0])}", f"dicom_seg_{int(nifti_seg.split('dicom_seg_')[1].split('.nii.gz')[0]) + j}"))))
        for nifti_file in nifti_files:
            if int(file.split('-')[-1]) < 8:
                if not('_2012' in nifti_file or '_2013' in nifti_file):
                    continue
            image = nib.load(nifti_file)

            if abs(abs(image.affine[0][0]) - abs(seg.affine[0][0])) > 0.00001:
                continue
            if int(file.split('-')[-1]) >= 8:
                if abs(abs(image.affine[1][0]) - abs(seg.affine[1][0])) > 0.00001:
                    continue

            if image.shape[:3] == seg.shape[:3]:
                shutil.copy(nifti_file, os.path.join(file, str(i), os.path.basename(nifti_file)))
        if int(file.split('-')[-1]) >= 8:
            copied_files = [os.path.join(file, str(i), f) for f in os.listdir(os.path.join(file, str(i))) if 'dicom_seg' not in f]
            if len(copied_files) > 1:
                if any('ADC' in os.path.basename(f) for f in copied_files):
                    for f in copied_files:
                        if 'ADC' not in f:
                            if os.path.exists(f):
                                os.remove(f)
                if any(int(f.split('_')[-1].split('.nii.gz')[0].replace('a','')) >= 100 for f in copied_files):
                    for f in copied_files:
                        if int(f.split('_')[-1].split('.nii.gz')[0].replace('a','')) < 100:
                            if os.path.exists(f):
                                os.remove(f)
            copied_files = [os.path.join(file, str(i), f) for f in os.listdir(os.path.join(file, str(i))) if 'dicom_seg' not in f]
            if len(copied_files) > 1:
                if int(file.split('-')[-1]) == 8:
                    dicom_file = [os.path.join(file, f) for f in os.listdir(os.path.join(file, str(i))) if 'dicom_seg' in f][0]
                    with open(os.path.join(file, 'log.txt'), 'r') as f:
                        lines = f.readlines()
                        dicom_source = [el.split('->')[0].rstrip() for el in lines if dicom_file in el][0]
                        if dicom_source.endswith('.569'):
                            if os.path.exists(os.path.join(file, str(i), 'Prostate_With_e-coil_C-_100a.nii.gz')):
                                os.remove(os.path.join(file, str(i), 'Prostate_With_e-coil_C-_100a.nii.gz'))
                        else:
                            if os.path.exists(os.path.join(file, str(i), 'Prostate_With_e-coil_C-_100.nii.gz')):
                                os.remove(os.path.join(file, str(i), 'Prostate_With_e-coil_C-_100.nii.gz'))

    for i in range(6):
        temp_dict = {}

        nifti_file = os.path.join(file, str(i), [el for el in os.listdir(os.path.join(file, str(i))) if 'dicom_seg' not in el][0])
        image = nib.load(nifti_file)
        nifti_segs = [os.path.join(file, str(i), el) for el in os.listdir(os.path.join(file, str(i))) if 'dicom_seg' in el]

        for nifti_seg in nifti_segs:
            ind = int(os.path.basename(nifti_seg).split("dicom_seg_")[1].split(".nii.gz")[0]) % 4
            label_name_local = label_names[ind]

            try:
                with open(os.path.join(file, "log.txt"), "r") as f:
                    lines = f.readlines()

                    seg_source = [
                        el.rstrip().split("->")[0]
                        for el in lines if os.path.basename(nifti_seg) in el
                    ]

                    image_source = [
                        el.rstrip().split("->")[0]
                        for el in lines if os.path.basename(nifti_file) in el
                    ]

                    temp_dict["image"] = image_source
                    temp_dict[label_name_local] = seg_source

            except FileNotFoundError:
                pass

            results.append(
                (nifti_file, nifti_seg, label_name_local, temp_dict)
            )

    return results
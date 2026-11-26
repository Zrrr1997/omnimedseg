from omnimedseg.core.paths import data_root
import os
from omnimedseg.utils.dataset_utils import *
from omnimedseg.converters.utils_ct import *
import SimpleITK as sitk

def convert_lnq_2023():
    converted_path = os.path.join(data_root(), 'CT', 'LNQ_2023', 'converted')
    label_classes = ['lymph_node']
    ensure_directories(converted_path, label_classes)

    compressed_base = os.path.join(data_root(), 'CT', 'LNQ_2023', 'compressed')
    extracted_base = os.path.join(data_root(), 'CT', 'LNQ_2023', 'extracted')
    #unzip_tcia(extracted_base, compressed_base)
    #convert_extracted_dicoms('LNQ_2023', extracted_base, converted_path)
    convert_dicom_segs('LNQ_2023', converted_path)

    match_and_save_niftis('LNQ_2023', converted_path, 'dicom_seg', None)



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
        img_file = [os.path.join(path, el) for el in os.listdir(path) if el.endswith('.nii.gz') and 'dicom_seg' not in el][0]

        cmd = [
            "segimage2itkimage",
            "--inputDICOM", dicom_seg_file,
            "--outputDirectory", os.path.dirname(dicom_seg_file),
            "-t", "nii"
        ]
        ds = pydicom.dcmread(dicom_seg_file)

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_file = os.path.join(path, 'log.txt')
        with open(log_file, 'r') as f:
            dicom_source = [el.split('->')[0].rstrip() for el in f.readlines() if dicom_seg_file in el][0]
            dst_path = os.path.join(path, f'dicom_seg_1.nii.gz')
            shutil.move(os.path.join(path, '1.nii.gz'), dst_path)
            resample_seg_to_image(dst_path, img_file, dst_path)

            with open(log_file, 'a+') as f:
                f.write(f"{dicom_source} -> {dst_path}\n")



    except Exception as e:
        print(f"❌ Failed to process DICOM SEG {path}: {e}")
        return None

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
    seg_binary = sitk.BinaryThreshold(seg_resampled, lowerThreshold=1, upperThreshold=65, insideValue=1, outsideValue=0)

    # Save output
    sitk.WriteImage(seg_binary, output_path)
CHAKSU_PATTERN = ['Test/1.0_Original', 'Train/1.0_Original', 'Test/3.0_Doctors', 'Train/3.0_Doctors', 'Test/5.0_OD_OC_Mean_Median_Majority_STAPLE/Bosch/Cup/Mean/', 'Test/5.0_OD_OC_Mean_Median_Majority_STAPLE/Forus/Cup/Mean/', 'Test/5.0_OD_OC_Mean_Median_Majority_STAPLE/Remidio/Cup/Mean/', 'Test/5.0_OD_OC_Mean_Median_Majority_STAPLE/Bosch/Disc/Mean/', 'Test/5.0_OD_OC_Mean_Median_Majority_STAPLE/Forus/Disc/Mean/', 'Test/5.0_OD_OC_Mean_Median_Majority_STAPLE/Remidio/Disc/Mean/', 'Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/Bosch/Cup/Mean/', 'Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/Forus/Cup/Mean/', 'Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/Remidio/Cup/Mean/', 'Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/Bosch/Disc/Mean/', 'Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/Forus/Disc/Mean/', 'Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/Remidio/Disc/Mean/']
# Datasets whose source can no longer be downloaded automatically. The user places the
# original archive(s) in datasets/<Modality>/<NAME>/ and the pipeline takes it from there.
MANUAL_DOWNLOADS = {
    "BUSI": (
        "The original BUSI download (https://scholar.cu.edu.eg/Dataset_BUSI.zip) is offline. "
        "Obtain Dataset_BUSI.zip (Al-Dhabyani et al., Data in Brief 28, 2020, "
        "doi:10.1016/j.dib.2019.104863) and place it at datasets/Ultrasound/BUSI/Dataset_BUSI.zip"
    ),
}

FILTER_PATTERNS = {
    'CHAKSU': CHAKSU_PATTERN,
}

FUNDUS_DATASET_URLS = {
    "PAPILA": "https://figshare.com/articles/dataset/PAPILA/14798004",
    "LES_AV": "https://figshare.com/articles/dataset/LES-AV_dataset/11857698",
    "FIVES": "https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169",
    "FUNDUS_AV_SEG": "https://figshare.com/articles/dataset/Fundus-AVSeg/27938034",
    "NETRA": "https://figshare.com/articles/dataset/NETRA_Dataset/22817144",
    "RETA": "https://figshare.com/articles/dataset/The_RETA_Benchmark_for_Retinal_Vascular_Tree_Analysis/16960855",
    "CHAKSU": "https://figshare.com/articles/dataset/Ch_k_u_A_glaucoma_specific_fundus_image_database/20123135",
    "HVDROPDB": "https://data.mendeley.com/public-files/datasets/xw5xc7xrmp/files/dfa3ec48-c763-439f-905a-bb0904ea840a/file_downloaded",
    "TREND": ["https://zenodo.org/records/4521044/files/1_HEALTHY%20YOUNG_RAW_GOOD_QUALITY.zip?download=1", "https://zenodo.org/records/4521044/files/2_HEALTHY%20YOUNG%20SEGMENTED.zip?download=1", "https://zenodo.org/records/4521044/files/4_HEALTHY%20OLD%20RAW_GOOD%20QUALITY.zip?download=1", "https://zenodo.org/records/4521044/files/5_HEALTHY%20OLD%20SEGMENTED.zip?download=1"],
    "CHASEDB": "https://researchinnovation.kingston.ac.uk/files/40659508/CHASEDB1.zip", #"https://researchdata.kingston.ac.uk/96/1/CHASEDB1.zip",
    "HRF": "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip",
    "CSC": "https://data.mendeley.com/public-api/zip/4k64fwnp4k/download/5",
    "AKRAM": "https://data.mendeley.com/public-api/zip/3csr652p9y/download/2",
    "ROP": "https://data.mendeley.com/public-api/zip/tv3b2bt6dy/download/2",
    "STARE": ["https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar", "https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar", "https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar"],
    "RAVIR": "https://drive.google.com/file/d/1ZlZoSStvE9VCRq3bJiGhQH931EF0h3hh/view",
    "ORVS": "https://github.com/AbdullahSarhan/ICPRVessels/tree/main/Vessels-Datasets/ORVS",
    "OCUTOX": "https://zenodo.org/api/records/13883485/files-archive",
    "COPH100": "https://doi.org/10.6084/m9.figshare/27061084"
}

ULTRASOUND_DATASET_URLS = {
    "BUSI": "https://scholar.cu.edu.eg/Dataset_BUSI.zip",
    "PSFHS": "https://zenodo.org/records/7851339/files/Pubic%20Symphysis-Fetal%20Head%20Segmentation%20and%20Angle%20of%20Progression.zip?download=1",
    "HC18": "https://zenodo.org/api/records/1327317/files-archive",
    "DDTI": "http://cimalab.unal.edu.co/applications/thyroid/thyroid.zip",
    "MMOTU": "https://figshare.com/articles/dataset/_zip/25058690",
    "MUSCLE_US": "https://data.mendeley.com/public-files/datasets/3jykz7wz8d/files/b1601a22-98a5-41ce-9fbb-54dd21834b55/file_downloaded",
    "TN3K": "https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view",
    "TG3K": "https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view",
    "100_US": ["https://www.researchgate.net/profile/Jan-Egger-2/publication/329586355_100_2D_US_Images_and_Tumor_Segmentation_Masks/data/5c10ddd4a6fdcc494fede23a/105-US-Images.zip?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIiwicG9zaXRpb24iOiJwYWdlQ29udGVudCJ9fQ", "https://www.researchgate.net/profile/Jan-Egger-2/publication/329586355_100_2D_US_Images_and_Tumor_Segmentation_Masks/data/5c10df354585157ac1bbfa72/105-US-Masks.zip?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIiwicG9zaXRpb24iOiJwYWdlQ29udGVudCJ9fQ"],
    "FASS": "https://data.mendeley.com/public-files/datasets/4gcpm9dsc3/files/89e74076-ff57-4e81-9634-4fc29c6128ff/file_downloaded",
    "BUS_BRA": "https://zenodo.org/api/records/8231412/files-archive",
    "BUS_UCLM": "https://data.mendeley.com/public-api/zip/7fvgj4jsp7/download/2",
}


ENDOSCOPY_DATASET_URLS = {
    'KVASIR_SEG': 'https://datasets.simula.no/downloads/kvasir-seg.zip',
    'KVASIR_CAPSULE_SEG': 'https://datasets.simula.no/downloads/kvasir-capsule-seg.zip',
    'POLYP_GEN': 'https://drive.google.com/drive/folders/16uL9n84SrMt7IiQFzTUQNaJ9TbHJ8DhW',
    'EAD': 'https://data.mendeley.com/public-files/datasets/c7fjbxcgj9/files/f07bd9ec-1416-4343-8314-8dee4b5efa71/file_downloaded',
    'FETOSCOPY_PLACENTA': 'https://rdr.ucl.ac.uk/articles/dataset/Fetoscopy_Placenta_Dataset/30536171',
    'FETREG': 'https://rdr.ucl.ac.uk/articles/dataset/_b_FetReg_Largescale_Multi-centre_Fetoscopy_Placenta_Dataset_b_/30417166',
    'POLYP_DB': 'https://osf.io/download/jg32z/',
    'CVC_CLINICDB': 'https://www.dropbox.com/scl/fi/ky766dwcxt9meq3aklkip/CVC-ClinicDB.rar?rlkey=61xclnrraadf1niqdvldlds93&e=1&dl=1',
    'HYPER_KVASIR': 'https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip',
    'ENDOVIS_15': ['https://archive.iar.kit.edu/opencas/data/endovis15_ins/Segmentation_Rigid_Training.zip', 'https://archive.iar.kit.edu/opencas/data/endovis15_ins/Segmentation_Rigid_Testing_Revision.zip', 'https://archive.iar.kit.edu/opencas/data/endovis15_ins/Segmentation_Rigid_Testing_GT.zip'],
    'LARYNGEAL_ENDOSCOPIC': 'https://github.com/imesluh/vocalfolds/archive/refs/heads/master.zip',
    'BAGLS': ['https://zenodo.org/records/3762320/files/test.zip?download=1', 'https://zenodo.org/records/3762320/files/training.zip?download=1']
}

OCT_DATASET_URLS = {
    'AMD_SD': 'https://springernature.figshare.com/articles/dataset/An_Optical_Coherence_Tomography_Image_Dataset_for_wet_AMD_Lesions_Segmentation/25513435',
    'OIMHS': 'https://springernature.figshare.com/articles/dataset/OIMHS_dataset/23508453',
    'AIDK': 'https://doi.org/10.6084/m9.figshare/25952845',
    'OCT_LESION': 'https://data.mendeley.com/public-files/datasets/btv6yrdbmv/files/72f260d2-ab9f-40bc-bac2-076699a964ba/file_downloaded',
    'INTRARETINAL_CYSTOID_FLUID': "https://datasetninja.com/intraretinal-cystoid-fluid"
}

MICROSCOPY_DATASET_URLS = {
    'SLIMIA': ['https://figshare.com/articles/dataset/OperaPhenix_zip/27390402', 'https://figshare.com/articles/dataset/OlympusIX05_zip/27390411', 'https://figshare.com/articles/dataset/Axiovert200M_zip/27393225', 'https://figshare.com/articles/dataset/Cytation5_zip/27393219', 'https://figshare.com/articles/dataset/LeicaDMi1_zip/27434550', 'https://figshare.com/articles/dataset/IncucyteS3_zip/27433092', 'https://figshare.com/articles/dataset/Axiovert200_zip/27433035', 'https://figshare.com/articles/dataset/LeicaDMi3000B_zip/27394764', 'https://figshare.com/articles/dataset/IncucyteZOOM_zip/27438526'],
    'PCMMD': 'https://data.mendeley.com/public-api/zip/3v2nrxpr9s/download/1',
    'BRIFISEG': ['https://zenodo.org/records/7195636/files/1channel.tar?download=1', 'https://zenodo.org/records/7195636/files/3channel.tar?download=1'],
    'WBC': 'https://github.com/zxaoyou/segmentation_WBC/archive/refs/heads/master.zip',
    'YEAZ': ['https://drive.google.com/uc?id=1Sot3bau0F0dsBjRxoQzdGOeUy_wMezal', 'https://drive.google.com/uc?id=14MUIN26ou0L12UC9UV_AC2S3isj1qBMY'],
    'BBBC010': ['https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip', 'https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground.zip'],
    'OCCISC': 'https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/Dataset.zip',
    'BBBC038': ['https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip', 'https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip', 'https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip'],
    'SSTEM': 'https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/refs/heads/master.zip',
    'BBBC041SEG':'https://drive.google.com/file/d/1nG-ra6BPAZSTsdYCvedzCo-JLD7jdH71/view',
    'CCAGT': 'https://data.mendeley.com/public-api/zip/wg4bpm33hj/download/2',
    'VICAR': 'https://zenodo.org/records/5153251/files/labelled.zip?download=1',
    'FLUORESCENT_NEURONAL_CELLS': ['https://amsacta.unibo.it/id/eprint/7347/28/green.zip', 'https://amsacta.unibo.it/id/eprint/7347/29/red.zip', 'https://amsacta.unibo.it/id/eprint/7347/27/yellow.zip'],
    'EMDS_6': 'https://figshare.com/articles/dataset/EMDS-6/17125025'
}

DERMOSCOPY_DATASET_URLS = {
    'HAM_10000': 'https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T&format=original',
    'ISIC_2016': ['https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip', 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip', 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Training_Data.zip', 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Training_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Test_Data.zip', 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part2B_Test_GroundTruth.zip'],
    'ISIC_2017': ['https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip', 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip', 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip', 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip'],
    'ISIC_2018': ['https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Validation_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip', 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Test_GroundTruth.zip'],
    'SKIN_HAIR': 'https://data.mendeley.com/public-api/zip/j5ywpd2p27/download/2',
    'NEVUS': 'https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzEzMjFfQWNjdXJhdGUgTmV2dXMgU2hhcGVzL2FjY3VyYXRlLW5ldnVzLXNoYXBlcy1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICJwb1RpL1QwaFhTbEVyMFdjSVVwUW5aWlpFVXVtaGF3RlZ4d2ZjWWxCUlJrPSJ9',
    'FUSC_2021': 'https://github.com/uwm-bigdata/wound-segmentation/archive/refs/heads/master.zip'
    }

XRAY_DATASET_URLS = {
    'PTX_498': 'https://zenodo.org/records/8266529/files/PTX-498-v2-fix.zip?download=1',
    'BTXRD': 'https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398',
    'PANDENTAL': 'https://data.mendeley.com/public-api/zip/hxt48yk462/download/1',
    'ARCADE': 'https://zenodo.org/records/10390295/files/arcade.zip?download=1',
    'HIPBONE': 'https://data.mendeley.com/public-api/zip/zm6bxzhmfz/download/1',
    'COVID19_CXR': 'https://github.com/GeneralBlockchain/covid-19-chest-xray-segmentations-dataset/archive/refs/heads/master.zip',
    'TEETH_SEG': 'https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzE1OThfVGVldGggU2VnbWVudGF0aW9uIG9uIERlbnRhbCBYLXJheSBJbWFnZXMvdGVldGgtc2VnbWVudGF0aW9uLW9uLWRlbnRhbC14LXJheS1pbWFnZXMtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAiRzlCdW93eXkvbnNLbGRkLzI2QVBEbC9DNnNSRklQMUFTN3hucFNUYTJ0OD0ifQ==',
}

CT_DATASET_URLS = {
    'ABDOMEN_1K': ['https://zenodo.org/records/5903099/files/AbdomenCT-1K-ImagePart1.zip?download=1', 'https://zenodo.org/records/5903846/files/AbdomenCT-1K-ImagePart2.zip?download=1', 'https://zenodo.org/records/5903769/files/AbdomenCT-1K-ImagePart3.zip?download=1', 'https://zenodo.org/records/5903769/files/Mask.7z?download=1'],
    'ADRENAL_ACC_KIT67': 'https://www.cancerimagingarchive.net/wp-content/uploads/Adrenal-ACC-Ki67-Seg_v1.tcia',
    'COVID19_LUNG_LESION': 'https://drive.google.com/drive/folders/1HXMjSV99v3BB6yJJoqsFhKvWe0LAKvt-',
    'COVID19_LUNG_AND_INFECTION': ['https://zenodo.org/records/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1', 'https://zenodo.org/records/3757476/files/Lung_and_Infection_Mask.zip?download=1'],
    'NSCLC_RADIOGENOMICS': 'https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC_Radiogenomics-6-1-21-Version-4.tcia',
    'NSCLC_RADIOMICS': 'https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-Version-4-Oct-2020-NBIA-manifest.tcia',
    'MSD_LUNG': 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar',
    'HCC_TACE_SEG': ['https://www.cancerimagingarchive.net/wp-content/uploads/HCC-TACE-Seg_v1_202201.tcia'],
    'LNQ_2023': 'https://www.cancerimagingarchive.net/wp-content/uploads/Mediastinal-Lymph-Node-SEG-DA-RAD.tcia',
    'MSD_LIVER': 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar',
    'MSD_COLON': 'https://drive.google.com/file/d/1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y/view?usp=drive_link',
    'MSD_PANCREAS': 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar',
    'MSD_SPLEEN': 'https://drive.google.com/file/d/1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE/view?usp=drive_link',
    'MSD_HEPATIC_VESSEL': 'https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view?usp=drive_link',
    'PLETHORA': ['https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-OriginalCTs.tcia', 'https://www.cancerimagingarchive.net/wp-content/uploads/PleThora-Thoracic_Cavities-June-2020.zip', 'https://www.cancerimagingarchive.net/wp-content/uploads/PleThora-Effusions-June-2020.zip'],
    'TOTAL_SEGMENTATOR': 'https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1',
    'WORD': 'https://drive.google.com/file/d/19OWCXZGrimafREhXm8O8w2HBHZTfxEgU/view',
    'CHAOS_CT': 'https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1',
    'FLARE_2021': ['https://zenodo.org/records/5903672/files/TrainingImg.zip?download=1', 'https://zenodo.org/records/5903672/files/TrainingMask.zip?download=1'],
    'FLARE_2022': 'https://zenodo.org/records/7860267/files/FLARE22Train.zip?download=1',
    'LUNA': ['https://zenodo.org/records/3723295/files/seg-lungs-LUNA16.zip?download=1', 'https://zenodo.org/records/3723295/files/subset0.zip?download=1', 'https://zenodo.org/records/3723295/files/subset2.zip?download=1', 'https://zenodo.org/records/3723295/files/subset3.zip?download=1', 'https://zenodo.org/records/3723295/files/subset4.zip?download=1', 'https://zenodo.org/records/3723295/files/subset5.zip?download=1', 'https://zenodo.org/records/3723295/files/subset6.zip?download=1' 'https://zenodo.org/records/4121926/files/subset7.zip?download=1', 'https://zenodo.org/records/4121926/files/subset8.zip?download=1', 'https://zenodo.org/records/4121926/files/subset9.zip?download=1'],
    'VERSE_2019': ['https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip', 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip', 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip'],
    'VERSE_2020': ['https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip', 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip', 'https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip'],
    'ATM': ['https://zenodo.org/records/7949582/files/TrainBatch1.rar?download=1', 'https://zenodo.org/records/6590774/files/TrainBatch2.rar?download=1'],
    'PANCREAS_CT': ['https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_pancreas_labels-02-05-2017-1.zip', 'https://www.cancerimagingarchive.net/wp-content/uploads/Pancreas-CT-20200910.tcia'],
    'ULS': ['https://zenodo.org/records/10035161/files/ULS23_Part1.z01?download=1','https://github.com/DIAGNijmegen/ULS23/archive/refs/heads/main.zip', 'https://zenodo.org/records/10035161/files/ULS23_Part1.z02?download=1', 'https://zenodo.org/records/10035161/files/ULS23_Part1.z03?download=1', 'https://zenodo.org/records/10035161/files/ULS23_Part1.zip?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z01?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z01?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z03?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z03?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z04?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z05?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z06?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z07?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z08?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z09?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.z10?download=1', 'https://zenodo.org/records/10050960/files/ULS23_Part2.zip?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z01?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z02?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z03?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z04?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z05?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z06?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z07?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z08?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.z09?download=1', 'https://zenodo.org/records/10054306/files/ULS23_Part3.zip?download=1'],
    'COLORECTAL_LIVER_METASTASES': 'https://www.cancerimagingarchive.net/wp-content/uploads/Colorectal-Liver-Metastases-November-2022-manifest.tcia',
    'AERO_PATH': 'https://zenodo.org/records/10069289/files/AeroPath.zip?download=1',
    'COUINAUD': 'https://github.com/GLCUnet/dataset/archive/refs/heads/master.zip',
    "DAP_ATLAS": "https://drive.google.com/file/d/1ex0a9eQULLvKPDwijmijX2h49A-ockNy/view",
    'FUMPE': ["https://figshare.com/articles/dataset/Ground_Truth/6265721", "https://figshare.com/articles/dataset/Patient35/6265685", "https://figshare.com/articles/dataset/Patient34/6265679", "https://figshare.com/articles/dataset/Patient33/6265670", "https://figshare.com/articles/dataset/Patient32/6265658", "https://figshare.com/articles/dataset/Patient31/6265655", "https://figshare.com/articles/dataset/Patient30/6265646", "https://figshare.com/articles/dataset/Patient29/6265643", "https://figshare.com/articles/dataset/Untitled_Item/6265640", "https://figshare.com/articles/dataset/Patient27/6265613", "https://figshare.com/articles/dataset/Patient26/6265604", "https://figshare.com/articles/dataset/Patient25/6265592", "https://figshare.com/articles/dataset/Patient24/6265589", "https://figshare.com/articles/dataset/Patient23/6265586", "https://figshare.com/articles/dataset/Patient22/6265583", "https://figshare.com/articles/dataset/Patient21/6265574", "https://figshare.com/articles/dataset/Patient20/6265562", "https://figshare.com/articles/dataset/Patient19/6265556", "https://figshare.com/articles/dataset/Patient18/6265544", "https://figshare.com/articles/dataset/Patient17/6265532", "https://figshare.com/articles/dataset/Patient16/6265526", "https://figshare.com/articles/dataset/Patient15/6265520", "https://figshare.com/articles/dataset/Patient14/6265511", "https://figshare.com/articles/dataset/Patient13/6265505", "https://figshare.com/articles/dataset/Patient12/6265502", "https://figshare.com/articles/dataset/Patient11/6265493", "https://figshare.com/articles/dataset/Patient09/6265466", "https://figshare.com/articles/dataset/Patient08/6265451", "https://figshare.com/articles/dataset/Patient07/6265433", "https://figshare.com/articles/dataset/Patient06/6265427", "https://figshare.com/articles/dataset/Patient05/6265379", "https://figshare.com/articles/dataset/Patient04/6265367", "https://figshare.com/articles/dataset/Patient03/6265364", "https://figshare.com/articles/dataset/Patient02/6265358", "https://figshare.com/articles/dataset/Patient01/6265292"],
    "PDDCA": ['https://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part1.zip', 'https://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part2.zip', 'https://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part3.zip', 'https://www.imagenglab.com/data/pddca/pddca.odt'],
    'RIBSEG_V2': ['https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view', 'https://zenodo.org/api/records/3893508/files-archive', 'https://zenodo.org/api/records/3893498/files-archive', 'https://zenodo.org/api/records/3893496/files-archive', 'https://zenodo.org/api/records/3993380/files-archive'],
    'TOPCOW': 'https://drive.switch.ch/index.php/s/rkqOO3adjmJVlMz/download',
    'SPINEMETS': 'https://www.cancerimagingarchive.net/wp-content/uploads/Spine-Mets-CT-SEG_v1_2024.tcia',
    'MEDIASTINAL_STRUCTURES': 'https://drive.google.com/uc?id=1ZsFq7PslqQ5ow_dXB01kDkaKPqYDXD5d',
    'CT_LYMPH_NODES': ['https://www.cancerimagingarchive.net/wp-content/uploads/MED_ABD_LYMPH_MASKS.zip', 'https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_CT_Lymph_Nodes_03-31-2023.tcia'],
    'PEDIATRIC_CT_SEG': 'https://www.cancerimagingarchive.net/wp-content/uploads/Pediatric-CT-SEG-Mar-22-2022-manifest.tcia',
    'CURVAS': 'https://zenodo.org/records/12687192/files/training_set.zip?download=1',
    'PROSTATE_EDGE_CASES': 'https://www.cancerimagingarchive.net/wp-content/uploads/Prostate-Anatomical-Edge-Cases-May-2023-manifest.tcia',
    'COVID19_20': 'https://drive.google.com/file/d/1WYyy67rXuT3jimOnD2PeFKwIT7efGp6n/view?usp=drive_link',
    'AUTOPET_2024_CT': 'https://fdat.uni-tuebingen.de/records/6gjsg-zcg93/files/psma-fdg-pet-ct-lesions.zip?download=1',
    'VESSEL12': ['https://zenodo.org/records/8055066/files/VESSEL12_01-05.tar.bz2?download=1', 'https://zenodo.org/records/8055066/files/VESSEL12_01-20_Lungmasks.tar.bz2?download=1', 'https://zenodo.org/records/8055066/files/VESSEL12_06-10.tar.bz2?download=1', 'https://zenodo.org/records/8055066/files/VESSEL12_11_15.tar.bz2?download=1', 'https://zenodo.org/records/8055066/files/VESSEL12_16-20.tar.bz2?download=1'],
    'STACOM_SLAWT_CT': ['https://figshare.com/ndownloader/files/5011837?private_link=2a5de3a2b89a3fb87932', 'https://figshare.com/ndownloader/files/5028424?private_link=0ffc9facc15436a41ff9'],
    'LCTSC': 'https://www.cancerimagingarchive.net/wp-content/uploads/LCTSC_v2_20190508.tcia',
    'PANCREATIC_CT_CBCT_SEG': 'https://www.cancerimagingarchive.net/wp-content/uploads/Pancreatic-CT-CBCT-SEG_v2_20220823.tcia',
    'STS_TOOTH_CT': ['https://zenodo.org/records/10597292/files/SD-Tooth.zip.001?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.002?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.003?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.004?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.005?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.006?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.007?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.008?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.009?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.010?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.011?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.012?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.013?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.014?download=1', 'https://zenodo.org/records/10597292/files/SD-Tooth.zip.015?download=1'],
    'WAW_TACE': ['https://zenodo.org/records/12741586/files/ct_scans_1_4_wawtace_09_05_24.zip?download=1', 'https://zenodo.org/records/12741586/files/ct_scans_2_4_wawtace_09_05_24.zip?download=1', 'https://zenodo.org/records/12741586/files/ct_scans_3_4_wawtace_09_05_24.zip?download=1', 'https://zenodo.org/records/12741586/files/ct_scans_4_4_wawtace_09_05_24.zip?download=1', 'https://zenodo.org/records/12741586/files/organ_masks_wawtace_09_05_2024.zip?download=1', 'https://zenodo.org/records/12741586/files/tumor_masks_wawtace_v1_08_05_2024.zip?download=1'],
    'PENGWIN': ['https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part1.zip?download=1', "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part2.zip?download=1", "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_labels.zip?download=1"],
    'KITS19': 'https://github.com/neheller/kits19/archive/refs/heads/master.zip',
    'KITS23': 'https://github.com/neheller/kits23/archive/refs/heads/main.zip',
    'LIVER_SEGMENTS': 'https://figshare.com/articles/dataset/MedSeg_Liver_segments_dataset/13643252',
    'MEDSEG_INFERIOR_VENA_CAVA': 'https://figshare.com/ndownloader/articles/15180192'
}

MRI_DATASET_URLS ={
    'ACDC': 'https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download',
    'CC_TUMOR_HETEROGENEITY': 'https://www.cancerimagingarchive.net/wp-content/uploads/CC-Tumor-Heterogeneity-manifest.tcia',
    'CHAOS_MRI': 'https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1',
    'CROSSMODA': 'https://www.cancerimagingarchive.net/wp-content/uploads/Vestibular-Schwannoma-SEG-Feb-2021-manifest.tcia',
    'ISLES_2022': 'https://zenodo.org/records/7960856/files/ISLES-2022.zip?download=1',
    'MSD_HEART': 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar',
    'MSD_PROSTATE': 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar',
    'NCI_ISBI': ['https://www.cancerimagingarchive.net/wp-content/uploads/NCI-ISBI-2013-Prostate-Challenge-Training.zip', 'https://www.cancerimagingarchive.net/wp-content/uploads/NCI-ISBI-2013-Prostate-Challenge-Leaderboard.zip', 'https://www.cancerimagingarchive.net/wp-content/uploads/NCI-ISBI-2013-Prostate-Challenge-Test.zip', 'https://www.cancerimagingarchive.net/wp-content/uploads/ISBI-Prostate-Challenge-Training.tcia', 'https://www.cancerimagingarchive.net/wp-content/uploads/ISBI-Prostate-Challenge-LeaderBoard.tcia', 'https://www.cancerimagingarchive.net/wp-content/uploads/ISBI-Prostate-Challenge-Testing.tcia'],
    'PICAI': ['https://zenodo.org/records/6624726/files/picai_public_images_fold0.zip?download=1', 'https://zenodo.org/records/6624726/files/picai_public_images_fold1.zip?download=1', 'https://zenodo.org/records/6624726/files/picai_public_images_fold2.zip?download=1', 'https://zenodo.org/records/6624726/files/picai_public_images_fold3.zip?download=1', 'https://zenodo.org/records/6624726/files/picai_public_images_fold4.zip?download=1', 'https://github.com/DIAGNijmegen/picai_labels/archive/refs/heads/main.zip'],
    'PROMISE12': ['https://zenodo.org/records/8026660/files/training_data.zip?download=1', 'https://zenodo.org/records/8026660/files/test_data.zip?download=1', 'https://zenodo.org/records/8026660/files/livechallenge_test_data.zip?download=1'],
    'QIN_PROSTATE': 'https://www.cancerimagingarchive.net/wp-content/uploads/QIN-PROSTATE-Repeatability_v2_20180510.tcia',
    'SPINE': ['http://www.cg.informatik.uni-siegen.de/data/Downloads/SpineDatasets.7z', 'http://www.cg.informatik.uni-siegen.de/data/Downloads/SpineSegmented.7z', 'http://www.cg.informatik.uni-siegen.de/data/Downloads/SpineSource.7z'],
    'CDEMRIS': 'https://figshare.com/articles/dataset/CDEMRIS_fibrosis_scar_challenge_data_2012/4214532',
    'SEGTHY': 'https://www.campar.in.tum.de/public_datasets/2022_plosone_eilers/MRI_data.zip',
    'ATRIASEG_2018': 'https://www.dropbox.com/scl/fi/nero2nlaocdcdfhzwu5h0/2018_UTAH_MICCAI.zip?rlkey=vkkfrkc2l6x1e61jqyutb35qn&dl=1',
    'HVSMR_20': 'https://figshare.com/ndownloader/articles/25226360',
    'SPIDER': ['https://zenodo.org/records/10159290/files/images.zip?download=1', 'https://zenodo.org/records/10159290/files/masks.zip?download=1'],
    'HNTSMRG': 'https://zenodo.org/records/11199559/files/HNTSMRG24_train.zip?download=1',
    'IBD': 'https://zenodo.org/records/13839321/files/A%20comprehensive%20dataset.zip?download=1',
    'HYPO_SUBFIELDS': 'https://plus.figshare.com/articles/dataset/A_paired_dataset_of_multi-modal_MRI_at_3_Tesla_and_7_Tesla_with_manual_hippocampal_subfield_segmentations_on_7T_T2-weighted_images/26075713',
    'TOM500': 'https://springernature.figshare.com/articles/dataset/TOM500_A_Multi-Organ_Annotated_Orbital_MRI_Dataset_for_Thyroid_Eye_Disease/27133389',
    'PITUITORY_TUMOR': 'https://springernature.figshare.com/ndownloader/articles/27894084',
    'RESECT': ['https://files.de-1.osf.io/v1/resources/jv8bk/providers/osfstorage/?zip=', 'https://ns9999k.webs.sigma2.no/10.11582_2017.00004/RESECT/'],
    'GBM_RESERVOIR': 'https://figshare.com/ndownloader/articles/28001450/versions/20',
    'MEDSEG_VENTRICLES': 'https://figshare.com/ndownloader/articles/19644636/1',
}

# Extract zips in separate folders to avoid labels overwriting images since they have the same filename
NEW_FOLDERS = [
    'VESSEL12',
    'COVID19_LUNG_AND_INFECTION',
    'LUNA',
    'PENGWIN',
    'MEDSEG_VENTRICLES',
    'PROMISE12',
    'MEDSEG_INFERIOR_VENA_CAVA'
]


MODALITY_MAP = {}
for key in list(FUNDUS_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'Fundus'
for key in list(ULTRASOUND_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'Ultrasound'
for key in list(ENDOSCOPY_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'Endoscopy'
for key in list(OCT_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'OCT'
for key in list(MICROSCOPY_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'Microscopy'
for key in list(DERMOSCOPY_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'Dermoscopy'
for key in list(XRAY_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'XRay'
for key in list(CT_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'CT'
for key in list(MRI_DATASET_URLS.keys()):
    MODALITY_MAP[key] = 'MRI'

DOWNLOAD_URLS = {
    'Fundus': FUNDUS_DATASET_URLS,
    'Ultrasound': ULTRASOUND_DATASET_URLS,
    'Endoscopy': ENDOSCOPY_DATASET_URLS,
    'OCT': OCT_DATASET_URLS,
    'Microscopy': MICROSCOPY_DATASET_URLS,
    'Dermoscopy': DERMOSCOPY_DATASET_URLS,
    'XRay': XRAY_DATASET_URLS,
    'CT': CT_DATASET_URLS,
    'MRI': MRI_DATASET_URLS
}


if __name__ == "__main__":
    pass

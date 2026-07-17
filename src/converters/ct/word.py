from src.converters.utils_ct import *

def convert_word():
    labels = [
        "liver",
        "spleen",
        "left_kidney",
        "right_kidney",
        "stomach",
        "gallbladder",
        "esophagus",
        "pancreas",
        "duodenum",
        "colon",
        "intestine",
        "adrenal",
        "rectum",
        "bladder",
        "left_femur_head",
        "right_femur_head"
    ]
    convert_msd('WORD', labels, 'WORD-V0.1.0')
    convert_msd('WORD', labels, 'WORD-V0.1.0', images_dir='imagesVal', labels_dir='labelsVal', start=100)
import os
from src.converters.ct import CONVERTERS
from src.converters.utils_ct import align_all_pairs

def convert_ct_dataset(name):
    name_upper = name.upper()
    if name_upper in CONVERTERS:
        print(f"🚀 Converting {name_upper}...")
        CONVERTERS[name_upper]()  # Calls the correct converter dynamically
        if name not in ['ADRENAL_ACC_KIT67']:
            align_all_pairs(os.path.join('datasets', 'CT', name_upper, 'converted'))
        
    else:
        print(f"❌ No converter found for dataset: {name}")


if __name__ == "__main__":
    pass



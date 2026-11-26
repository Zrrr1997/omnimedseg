import os
from omnimedseg.converters.oct import CONVERTERS

def convert_oct_dataset(name):
    name_upper = name.upper()
    if name_upper in CONVERTERS:
        print(f"🚀 Converting {name_upper}...")
        CONVERTERS[name_upper]()  # Calls the correct converter dynamically
    else:
        print(f"❌ No converter found for dataset: {name}")


if __name__ == "__main__":
    pass
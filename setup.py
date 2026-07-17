#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="omnimedseg",
    version="1.0.0",
    author="Your Name",
    description="Medical imaging dataset download and conversion tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Changed from find_packages(where="src")
    package_dir={"": "."},      # Changed to look in current directory
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
        "pydicom>=2.3.0",
        "scikit-image>=0.19.0",
        "scipy>=1.7.0",
        "gdown>=4.0.0",
        "beautifulsoup4>=4.12.0",
        "undetected-chromedriver>=3.0.0",
        "dcm2niix>=1.0.20230310",
        "simpleitk>2.5.0",
        "xmltodict>=0.14.2",
        "h5py>=3.13.0",
        "nibabel>=5.3.2",
        "pandas>=2.2.3",
        "pynrrd>=1.1.3",
        "platipy>=0.7.2",
        "tcia_utils>=2.4.20",
        "dcmqi>=1.5.6",
        "isic-cli>=12.4.0"
    ],
    entry_points={
        "console_scripts": [
            "omnimedseg=src.cli.main:main",
        ],
    },
)
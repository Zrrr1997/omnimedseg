"""OASBUD: Open Access Series of Breast Ultrasonic Data (Piotrzkowska-Wróblewska et al., Med. Phys. 2017).

TEMPLATE: not implemented yet. OASBUD ships a single MATLAB file (OASBUD.mat) with *raw
radio-frequency (RF) echo signals* of breast lesions plus lesion ROI masks, not ready-made
B-mode images. A converter therefore has to reconstruct images before writing the standard format.

Suggested steps (check the field names against the actual .mat file and the paper):
  1. Load OASBUD.mat (scipy.io.loadmat) from <data_root>/Ultrasound/OASBUD/
  2. For every lesion and scan plane: envelope detection of the RF lines (abs of the Hilbert
     transform), log compression to a fixed dynamic range (e.g. 50 dB), scale to uint8
  3. Write the B-mode image to converted/images/000000.png and the ROI mask (0/255) to
     converted/labels/<label_name>/000000.png using ensure_directories(...)
  4. Record the source (file + record index + plane) per output file and call save_mapping(...)
  5. Add OASBUD to the golden tests once the output is final

A converted version produced earlier is part of the RADAR4KIT Ultrasound package
(doi:10.35097/qccrp1pkbk1j0dq0); its metadata.json documents the expected labels.
"""

import os

from omnimedseg.core.paths import data_root


def convert_oasbud():
    raw = os.path.join(data_root(), 'Ultrasound', 'OASBUD')
    raise NotImplementedError(
        "The OASBUD converter is a template and not implemented yet; see "
        "src/omnimedseg/converters/ultrasound/oasbud.py. "
        f"The downloaded raw data (if any) is kept in {raw}. A converted version is available on "
        "RADAR4KIT: https://doi.org/10.35097/qccrp1pkbk1j0dq0"
    )

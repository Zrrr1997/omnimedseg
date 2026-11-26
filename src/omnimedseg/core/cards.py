"""Dataset cards: one YAML file per dataset, the single source of truth for dataset metadata.

Cards live in ``omnimedseg/cards/<Modality>/<NAME>.yaml``. Everything the pipeline needs to know
about a dataset (where to download it, how to extract it, its license and citation) is read from
there; the converter code is looked up separately by name (see ``omnimedseg.converters.registry``).
"""

from functools import lru_cache
from importlib import resources
from typing import Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omnimedseg.core.errors import OmniMedSegError

MODALITIES = ("CT", "Dermoscopy", "Endoscopy", "Fundus", "Microscopy", "MRI", "OCT", "Ultrasound", "XRay")
Modality = Literal["CT", "Dermoscopy", "Endoscopy", "Fundus", "Microscopy", "MRI", "OCT", "Ultrasound", "XRay"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Download(_Strict):
    method: Literal["auto", "manual"] = "auto"
    urls: List[str] = Field(min_length=1)
    instructions: Optional[str] = None  # required for manual downloads

    @model_validator(mode="after")
    def _manual_needs_instructions(self):
        if self.method == "manual" and not self.instructions:
            raise ValueError("manual downloads need 'instructions'")
        return self

    @property
    def url_arg(self) -> Union[str, List[str]]:
        """The URL argument the downloaders expect: a string for one URL, a list for several."""
        return self.urls[0] if len(self.urls) == 1 else list(self.urls)


class Extract(_Strict):
    filter_patterns: List[str] = []  # only extract archive members matching these paths
    new_folder: bool = False  # extract each archive into its own folder (avoids filename clashes)


class License(_Strict):
    spdx: Optional[str] = None  # SPDX identifier/expression; null if the license is not a standard one
    text: Optional[str] = None  # verbatim license statement from the RADAR4KIT rights statement
    non_commercial: Optional[bool] = None  # True if commercial use is restricted; null if unknown


class Source(_Strict):
    homepage: Optional[str] = None
    citation: Optional[str] = None  # BibTeX


class DatasetCard(_Strict):
    name: str = Field(pattern=r"^[A-Z0-9][A-Z0-9_]*$")
    modality: Modality
    download: Download
    extract: Extract = Extract()
    license: License = License()
    source: Source = Source()
    radar_doi: Optional[str] = Field(default=None, pattern=r"^10\.\d+/\S+$")  # converted package on RADAR4KIT
    depends_on: List[str] = []  # datasets that must be downloaded and extracted first
    notes: Optional[str] = None


class CardError(OmniMedSegError):
    """A dataset card is missing or invalid."""


class UnknownDatasetError(CardError, LookupError):
    """No card exists for the requested dataset name."""


def _card_files():
    root = resources.files("omnimedseg") / "cards"
    for modality_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not modality_dir.is_dir() or modality_dir.name.startswith(("_", ".")):
            continue
        for f in sorted(modality_dir.iterdir(), key=lambda p: p.name):
            if f.name.endswith(".yaml"):
                yield modality_dir.name, f


@lru_cache(maxsize=None)
def load_cards() -> Dict[str, DatasetCard]:
    """Load and validate all cards. Returns {NAME: card}, sorted by modality then name."""
    cards: Dict[str, DatasetCard] = {}
    for folder, f in _card_files():
        where = f"cards/{folder}/{f.name}"
        try:
            card = DatasetCard.model_validate(yaml.safe_load(f.read_text(encoding="utf-8")))
        except (ValidationError, yaml.YAMLError) as e:
            raise CardError(f"{where}: {e}") from e
        if f.name != f"{card.name}.yaml":
            raise CardError(f"{where}: file name must be {card.name}.yaml")
        if folder != card.modality:
            raise CardError(f"{where}: card says modality {card.modality} but is in folder {folder}/")
        if card.name in cards:
            raise CardError(f"{where}: duplicate dataset name {card.name}")
        cards[card.name] = card
    for card in cards.values():
        missing = [d for d in card.depends_on if d not in cards]
        if missing:
            raise CardError(f"cards/{card.modality}/{card.name}.yaml: depends_on unknown datasets {missing}")
    return cards


def get_card(name: str) -> DatasetCard:
    try:
        return load_cards()[name]
    except KeyError:
        raise UnknownDatasetError(f"Unknown dataset '{name}'") from None


def datasets_for_modality(modality: str) -> List[str]:
    return [n for n, c in load_cards().items() if c.modality == modality]

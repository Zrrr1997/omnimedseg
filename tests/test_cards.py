"""Dataset cards: schema, consistency, and the metadata the community and challenges rely on."""

import pytest

from omnimedseg.core.cards import MODALITIES, DatasetCard, get_card, load_cards

CARDS = load_cards()

# Known gaps, tracked so they are not forgotten. Remove names from here as they are filled in.
KNOWN_MISSING_LICENSE = {"POLYP_GEN"}  # not part of the RADAR4KIT release
KNOWN_MISSING_RADAR = {"POLYP_GEN"}


def test_all_cards_load():
    assert len(CARDS) == 155


def test_every_modality_has_cards():
    assert {c.modality for c in CARDS.values()} == set(MODALITIES)


@pytest.mark.parametrize("name", sorted(CARDS))
def test_card_has_license_and_radar_record(name):
    card = CARDS[name]
    if name not in KNOWN_MISSING_LICENSE:
        assert card.license.text, f"{name}: no license statement"
    if name not in KNOWN_MISSING_RADAR:
        assert card.radar_doi, f"{name}: no RADAR4KIT DOI"


def test_known_gaps_are_still_gaps():
    """Fails once a known gap is fixed, as a reminder to remove it from the lists above."""
    for name in KNOWN_MISSING_LICENSE:
        assert not CARDS[name].license.text, f"{name} now has a license; remove it from KNOWN_MISSING_LICENSE"
    for name in KNOWN_MISSING_RADAR:
        assert not CARDS[name].radar_doi, f"{name} now has a DOI; remove it from KNOWN_MISSING_RADAR"


def test_spdx_and_non_commercial_agree():
    for card in CARDS.values():
        if card.license.spdx:
            assert card.license.non_commercial == ("-NC" in card.license.spdx), card.name


def test_manual_download_needs_instructions():
    with pytest.raises(ValueError):
        DatasetCard.model_validate({"name": "X", "modality": "CT", "download": {"method": "manual", "urls": ["u"]}})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValueError):
        DatasetCard.model_validate({"name": "X", "modality": "CT", "download": {"urls": ["u"]}, "licence": {}})


def test_url_arg_shape():
    assert isinstance(get_card("HRF").download.url_arg, str)
    assert isinstance(get_card("STARE").download.url_arg, list)


def test_unknown_dataset_is_a_lookup_error():
    with pytest.raises(LookupError):
        get_card("NOT_A_DATASET")


def test_dependencies():
    assert get_card("DAP_ATLAS").depends_on == ["AUTOPET_2024_CT"]
    assert get_card("COUINAUD").depends_on == ["MSD_HEPATIC_VESSEL"]

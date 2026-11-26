"""Pipeline logic for dataset processing.

This module orchestrates the entire dataset processing workflow:
download, extraction, conversion, and validation.
"""

from omnimedseg.core.paths import data_root
import os
import shutil
import logging
from typing import Dict, List

from omnimedseg.core.cards import get_card
from omnimedseg.core.errors import DownloadError, ManualDownloadRequired, PipelineError
from omnimedseg.converters.registry import convert_dataset
from omnimedseg.downloaders.download_datasets import download_dataset, extract_all_zips
from omnimedseg.utils.dataset_utils import check_filename_map, remove_raw_data, filter_empty_labels

logger = logging.getLogger(__name__)

# Sub-folders of a dataset directory that are pipeline output, not downloaded raw data.
_OUTPUT_DIRS = {"converted", "converted.failed"}


def _has_raw_files(dataset_dir: str) -> bool:
    """True if ``dataset_dir`` holds at least one non-empty downloaded file (outside pipeline output).

    Note: ``*.part`` files count — gdown leaves complete archives under a ``.part`` name when its
    output path is a directory (e.g. MSD_SPLEEN), and extraction detects them by content.
    """
    if not os.path.isdir(dataset_dir):
        return False
    for root, dirs, files in os.walk(dataset_dir):
        if root == dataset_dir:
            dirs[:] = [d for d in dirs if d not in _OUTPUT_DIRS]
        for f in files:
            if os.path.getsize(os.path.join(root, f)) > 0:
                return True
    return False


def _download(dataset: str, modality: str, filter_pattern, new_folder: bool) -> None:
    dataset_dir = os.path.join(data_root(), modality, dataset)
    card = get_card(dataset)

    if card.download.method == "manual":
        if _has_raw_files(dataset_dir):
            logger.info(f"📁 {dataset} requires manual download; using the files found in {dataset_dir}")
            extract_all_zips(dataset_dir, dataset, filter_pattern=filter_pattern, new_folder=new_folder)
            return
        raise ManualDownloadRequired(
            f"{dataset} cannot be downloaded automatically. {card.download.instructions} "
            f"(with the current data root, the folder is: {os.path.abspath(dataset_dir)})"
        )

    logger.info(f"📥 Downloading {dataset}...")
    download_dataset(
        dataset,
        card.download.url_arg,
        modality,
        filter_pattern=filter_pattern,
        new_folder=new_folder
    )
    if not _has_raw_files(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, 'converted')):
            logger.warning(f"⚠️  No raw files for {dataset}, but a converted version exists; continuing.")
            return
        raise DownloadError(
            f"Download of {dataset} produced no files in {dataset_dir}. "
            f"Check the log above; the source URL may have changed: {', '.join(card.download.urls)}"
        )


def _convert(dataset: str, modality: str, filter_pattern, new_folder: bool,
             force_convert: bool, keep_raw_data: bool) -> None:
    dataset_dir = os.path.join(data_root(), modality, dataset)
    converted_path = os.path.join(dataset_dir, 'converted')
    failed_path = os.path.join(dataset_dir, 'converted.failed')

    if os.path.exists(converted_path) and not force_convert:
        logger.info(f"⏭️  {dataset} already converted. Use --force_convert to override.")
        return

    # Ensure raw data exists before conversion
    extracted_path = os.path.join(dataset_dir, 'extracted')
    compressed_path = os.path.join(dataset_dir, 'compressed')
    if not os.path.exists(extracted_path) and not os.path.exists(compressed_path):
        logger.info(f"📦 Extracting {dataset} before conversion...")
        extract_all_zips(dataset_dir, dataset, filter_pattern=filter_pattern, new_folder=new_folder)

    logger.info(f"🔄 Converting {dataset}...")
    try:
        convert_dataset(modality, dataset)
    except BaseException:
        # Never leave half-written output where it looks like a finished conversion.
        if os.path.exists(converted_path):
            shutil.rmtree(failed_path, ignore_errors=True)
            os.rename(converted_path, failed_path)
            logger.error(f"❌ Conversion of {dataset} failed; partial output moved to {failed_path}")
        raise

    shutil.rmtree(failed_path, ignore_errors=True)
    if not keep_raw_data:
        remove_raw_data(dataset, modality)


def process_datasets(
    dataset_names: List[str],
    modality: str,
    download: bool = False,
    convert: bool = False,
    extract: bool = False,
    delete_raw_data: bool = False,
    force_convert: bool = False,
    keep_raw_data: bool = False,
    filter_empty: bool = False,
    check_integrity: bool = False
) -> None:
    """
    Process one or more datasets.

    Each dataset is processed independently: a failure is logged and the remaining
    datasets still run. Raw data is only removed after a successful conversion.

    Args:
        dataset_names: List of dataset names to process
        modality: Modality name (e.g., 'Fundus', 'CT')
        download: Whether to download the dataset
        convert: Whether to convert the dataset
        extract: Whether to extract the dataset
        delete_raw_data: Whether to delete raw data after processing
        force_convert: Whether to force conversion even if already converted
        keep_raw_data: Whether to keep raw data after conversion
        filter_empty: Whether to filter out samples with empty labels
        check_integrity: Whether to check dataset integrity

    Raises:
        PipelineError: If one or more datasets failed (after all datasets were attempted)
        KeyboardInterrupt: If user interrupts the process
    """
    failures: Dict[str, BaseException] = {}

    for dataset in dataset_names:
        dataset_dir = os.path.join(data_root(), modality, dataset)
        logger.info(f"Processing dataset: {dataset} (modality: {modality})")
        try:
            card = get_card(dataset)
            filter_pattern = list(card.extract.filter_patterns) or None
            new_folder = card.extract.new_folder

            # 1. Download
            if download:
                _download(dataset, modality, filter_pattern, new_folder)

            # 2. Extract
            if extract:
                logger.info(f"📦 Extracting {dataset}...")
                extract_all_zips(dataset_dir, dataset, filter_pattern=filter_pattern, new_folder=new_folder)

            # 3. Convert
            if convert:
                _convert(dataset, modality, filter_pattern, new_folder, force_convert, keep_raw_data)

            # 4. Delete raw data
            if delete_raw_data:
                logger.info(f"🗑️  Deleting raw data for {dataset}...")
                remove_raw_data(dataset, modality)

            # 5. Filter empty labels
            if filter_empty:
                logger.info(f"🔍 Filtering empty labels for {dataset}...")
                filter_empty_labels(dataset, modality)

            # 6. Check integrity
            if check_integrity:
                extracted_path = os.path.join(dataset_dir, 'extracted')
                if not os.path.exists(extracted_path):
                    logger.warning(
                        f"⚠️  Please extract {dataset} first with:\n"
                        f"   omnimedseg extract {dataset}"
                    )
                else:
                    logger.info(f"✅ Checking integrity of {dataset}...")
                    check_filename_map(dataset, modality)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"❌ {dataset} failed: {e}")
            failures[dataset] = e
            continue

        logger.info(f"✅ Finished processing {dataset}")

    if failures:
        raise PipelineError(failures)

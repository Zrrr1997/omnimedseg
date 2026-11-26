"""Pipeline logic for dataset processing.

This module orchestrates the entire dataset processing workflow:
download, extraction, conversion, and validation.
"""

import os
import logging
from typing import List, Optional, Tuple

from omnimedseg.core.constants import (
    MODALITY_MAP, DOWNLOAD_URLS, CONVERTERS,
    FILTER_PATTERNS, NEW_FOLDERS
)
from omnimedseg.downloaders.download_datasets import download_dataset, extract_all_zips
from omnimedseg.utils.dataset_utils import check_filename_map, remove_raw_data, filter_empty_labels

logger = logging.getLogger(__name__)


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
        ValueError: If dataset or modality is not found
        KeyboardInterrupt: If user interrupts the process
        Exception: For any other processing errors
    """
    for dataset in dataset_names:
        # Get dataset-specific settings
        filter_pattern = FILTER_PATTERNS.get(dataset)
        new_folder = dataset in NEW_FOLDERS
        
        logger.info(f"Processing dataset: {dataset} (modality: {modality})")
        
        # 1. Download
        if download:
            logger.info(f"📥 Downloading {dataset}...")
            download_dataset(
                dataset,
                DOWNLOAD_URLS[modality][dataset],
                modality,
                filter_pattern=filter_pattern,
                new_folder=new_folder
            )
        
        # 2. Extract
        if extract:
            logger.info(f"📦 Extracting {dataset}...")
            dest_folder = os.path.join('datasets', modality, dataset)
            extract_all_zips(dest_folder, dataset, filter_pattern=filter_pattern, new_folder=new_folder)
        
        # 3. Convert
        if convert:
            converted_path = os.path.join('datasets', modality, dataset, 'converted')
            if os.path.exists(converted_path) and not force_convert:
                logger.info(f"⏭️  {dataset} already converted. Use --force_convert to override.")
            else:
                # Ensure raw data exists before conversion
                extracted_path = os.path.join('datasets', modality, dataset, 'extracted')
                compressed_path = os.path.join('datasets', modality, dataset, 'compressed')
                
                if not os.path.exists(extracted_path) and not os.path.exists(compressed_path):
                    logger.info(f"📦 Extracting {dataset} before conversion...")
                    dest_folder = os.path.join('datasets', modality, dataset)
                    extract_all_zips(dest_folder, dataset, filter_pattern=filter_pattern, new_folder=new_folder)
                
                logger.info(f"🔄 Converting {dataset}...")
                converter = CONVERTERS[modality]
                converter(dataset)
                
                if not keep_raw_data:
                    remove_raw_data(dataset, modality)
        
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
            extracted_path = os.path.join('datasets', modality, dataset, 'extracted')
            if not os.path.exists(extracted_path):
                logger.warning(
                    f"⚠️  Please extract {dataset} first with:\n"
                    f"   omnimedseg --dataset {dataset} --extract"
                )
            else:
                logger.info(f"✅ Checking integrity of {dataset}...")
                check_filename_map(dataset, modality)
        
        logger.info(f"✅ Finished processing {dataset}")


def resolve_datasets(
    dataset_name: Optional[str],
    modality: str
) -> Tuple[List[str], str]:
    """
    Resolve which datasets to process based on input arguments.

    Args:
        dataset_name: Single dataset name or None
        modality: Modality name

    Returns:
        Tuple containing:
            - List of dataset names
            - Modality name

    Raises:
        ValueError: If dataset or modality is not found
    """
    if dataset_name is not None:
        if dataset_name not in MODALITY_MAP:
            raise ValueError(f"Dataset '{dataset_name}' not found in MODALITY_MAP")
        return [dataset_name], MODALITY_MAP[dataset_name]
    
    if modality not in DOWNLOAD_URLS:
        raise ValueError(f"Modality '{modality}' not found")
    
    datasets = list(DOWNLOAD_URLS[modality].keys())
    return datasets, modality

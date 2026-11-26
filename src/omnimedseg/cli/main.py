#!/usr/bin/env python
"""Command-line interface for dataset download and conversion."""

import argparse
import sys
import logging
import os

from omnimedseg.core.pipeline import process_datasets, resolve_datasets
from omnimedseg.core.constants import DOWNLOAD_URLS
from omnimedseg.core.errors import PipelineError
from omnimedseg.utils.logger import setup_logging, get_logger
from omnimedseg.utils.banner import print_banner

logger = get_logger(__name__)

# Older names still accepted on the command line.
MODALITY_ALIASES = {'Ultrasound_2D': 'Ultrasound'}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process medical imaging datasets - download, extract, and convert to standardized format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Download and convert a single dataset
        omnimedseg --dataset PAPILA --download --convert

        # Process all datasets in a modality
        omnimedseg --modality Fundus --download --convert

        # Extract only (no conversion)
        omnimedseg --dataset PAPILA --extract

        # Convert without re-downloading
        omnimedseg --dataset PAPILA --convert --force_convert
        
        # Browse datasets
        omnimedseg --visualization
        omnimedseg --visualization --port 8080 --max_images 20
        
        # Train a model (config automatically saved to runs/run_name/logs/config.json)
        omnimedseg --train --modalities Fundus --datasets PAPILA --label_classes optic_cup
        
        # Train with custom image size
        omnimedseg --train --modalities Fundus --datasets PAPILA --label_classes optic_cup --image_size 512
        
        # Train with custom run name
        omnimedseg --train --run_name papila_optic_cup_v1 --modalities Fundus --datasets PAPILA
        
        # Predict on validation set with overlay
        omnimedseg --predict_val --load_checkpoint runs/run_name/checkpoints/best_model.pth --modalities Fundus --datasets PAPILA
        
        # Predict with custom overlay settings
        omnimedseg --predict_val --load_checkpoint runs/run_name/checkpoints/best_model.pth --modalities Fundus --datasets PAPILA --overlay_color red --overlay_alpha 0.3
        
        # Predict without overlay
        omnimedseg --predict_val --load_checkpoint runs/run_name/checkpoints/best_model.pth --modalities Fundus --datasets PAPILA --no_overlay
        
        # List all runs
        omnimedseg --list_runs
                """
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset name (e.g. NETRA). If not set, processes all datasets from --modality.'
    )
    parser.add_argument(
        '--modality', type=str, default='Fundus',
        choices=sorted(DOWNLOAD_URLS) + sorted(MODALITY_ALIASES),
        help='Modality name (default: Fundus)'
    )
    
    # Actions
    parser.add_argument(
        '--download', action='store_true',
        help='Download dataset(s)'
    )
    parser.add_argument(
        '--extract', action='store_true',
        help='Extract downloaded dataset(s)'
    )
    parser.add_argument(
        '--convert', action='store_true',
        help='Convert dataset(s) to standardized format'
    )
    parser.add_argument(
        '--check_integrity', action='store_true',
        help='Check if all files are downloaded, extracted, and converted correctly'
    )

    # Visualization/Browse
    parser.add_argument(
        '--visualization', action='store_true',
        help='Start dataset visualization tool'
    )
    
    # Browser options (only used with --visualization)
    parser.add_argument(
        '--root', type=str, default='datasets_to_review',
        help='Root directory containing datasets (default: datasets_to_review)'
    )
    parser.add_argument(
        '--size', type=int, default=512,
        help='Image size for display (default: 512)'
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--max_images', type=int, default=10,
        help='Maximum number of images to load per dataset (default: 10)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of parallel workers for loading (default: 8)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode for the browser'
    )
    
    # Training options
    parser.add_argument(
        '--train', action='store_true',
        help='Train a segmentation model'
    )
    parser.add_argument(
        '--predict', type=str,
        help='Predict segmentation for an image or directory'
    )
    parser.add_argument(
        '--predict_val', action='store_true',  
        help='Predict on the validation set using saved split info'
    )
    parser.add_argument(
        '--load_checkpoint', type=str,
        help='Load checkpoint for prediction or resume training'
    )
    parser.add_argument(
        '--modalities', type=str, nargs='+',
        help='Modalities to use for training'
    )
    parser.add_argument(
        '--datasets', type=str, nargs='+',
        help='Specific datasets to train on'
    )
    parser.add_argument(
        '--label_classes', type=str, nargs='+',
        help='Label classes to include'
    )
    parser.add_argument(
        '--exclude_datasets', type=str, nargs='+',
        help='Datasets to exclude'
    )
    parser.add_argument(
        '--exclude_classes', type=str, nargs='+',
        help='Label classes to exclude'
    )
    parser.add_argument(
        '--max_samples_per_dataset', type=int, default=None,
        help='Maximum samples per dataset'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--features', type=int, nargs='+', default=[64, 128, 256, 512],
        help='Features per layer for the model'
    )
    parser.add_argument(
        '--image_size', type=int, default=256,  # <-- ADD THIS
        help='Input image size for training (default: 256)'
    )
    parser.add_argument(
        '--bilinear', action='store_true', default=True,
        help='Use bilinear upsampling'
    )
    parser.add_argument(
        '--run_name', type=str, default=None,
        help='Custom run name (auto-generated if not provided)'
    )
    parser.add_argument(
        '--base_dir', type=str, default='runs',
        help='Base directory for training runs (default: runs)'
    )
    parser.add_argument(
        '--list_runs', action='store_true',
        help='List all available training runs'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--no_mixed_precision', action='store_true',
        help='Disable mixed precision training'
    )
    
    # Overlay options
    parser.add_argument(
        '--overlay', action='store_true', default=True,
        help='Save overlay images with predictions (default: True)'
    )
    parser.add_argument(
        '--no_overlay', action='store_true',
        help='Disable overlay images'
    )
    parser.add_argument(
        '--overlay_alpha', type=float, default=0.5,
        help='Transparency of overlay (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--overlay_color', type=str, default='green',
        choices=['green', 'red', 'blue', 'yellow', 'magenta', 'cyan', 'white'],
        help='Overlay color: green, red, blue, yellow, magenta, cyan, white (default: green)'
    )
    
    # Options
    parser.add_argument(
        '--force_convert', action='store_true',
        help='Force conversion even if already converted'
    )
    parser.add_argument(
        '--keep_raw_data', action='store_true',
        help='Keep raw data after conversion'
    )
    parser.add_argument(
        '--delete_raw_data', action='store_true',
        help='Delete raw data (extracted files)'
    )
    parser.add_argument(
        '--filter_empty_labels', action='store_true',
        help='Filter out samples with empty labels'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress all output except errors'
    )
    
    # Add flag to disable banner
    parser.add_argument(
        '--no-banner', action='store_true',
        help='Suppress the ASCII banner'
    )
    
    args = parser.parse_args()
    args.modality = MODALITY_ALIASES.get(args.modality, args.modality)
    return args


def main() -> None:
    """Main entry point for the CLI."""
    print_banner()

    args = parse_args()

    # Setup logging first (so we can log if needed)
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Handle visualization/browse mode
    if args.visualization:
        try:
            from omnimedseg.utils.visualize import start_visualization
            
            print("\n" + "=" * 60)
            print("🚀 Starting Dataset Visualization Tool")
            print("=" * 60)
            
            start_visualization(
                root=args.root,
                size=args.size,
                host=args.host,
                port=args.port,
                debug=args.debug,
                max_images=args.max_images,
                workers=args.workers
            )
        except ImportError as e:
            print(f"❌ Error: Could not import visualization module: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error starting browser: {e}")
            sys.exit(1)
        return
    
    # Handle training/prediction
    if args.train or args.predict or args.predict_val or args.list_runs:
        try:
            from omnimedseg.models.train import main as train_main
            
            # Build sys.argv for the training CLI
            train_argv = [sys.argv[0]]
            
            # Basic actions
            if args.train:
                train_argv.append('--train')
            if args.predict:
                train_argv.extend(['--predict', args.predict])
            if args.predict_val:
                train_argv.append('--predict_val')
            if args.load_checkpoint:
                train_argv.extend(['--load_checkpoint', args.load_checkpoint])
            if args.list_runs:
                train_argv.append('--list_runs')
            
            # Data selection
            if args.modalities:
                train_argv.extend(['--modalities'] + args.modalities)
            if args.datasets:
                train_argv.extend(['--datasets'] + args.datasets)
            if args.label_classes:
                train_argv.extend(['--label_classes'] + args.label_classes)
            if args.exclude_datasets:
                train_argv.extend(['--exclude_datasets'] + args.exclude_datasets)
            if args.exclude_classes:
                train_argv.extend(['--exclude_classes'] + args.exclude_classes)
            if args.max_samples_per_dataset:
                train_argv.extend(['--max_samples_per_dataset', str(args.max_samples_per_dataset)])
            
            # Training parameters
            if args.epochs:
                train_argv.extend(['--epochs', str(args.epochs)])
            if args.batch_size:
                train_argv.extend(['--batch_size', str(args.batch_size)])
            if args.lr:
                train_argv.extend(['--lr', str(args.lr)])
            if args.val_split:
                train_argv.extend(['--val_split', str(args.val_split)])
            
            # Model parameters
            if args.features:
                train_argv.extend(['--features'] + [str(f) for f in args.features])
            if args.image_size:  
                train_argv.extend(['--image_size', str(args.image_size)])
            if args.bilinear:
                train_argv.append('--bilinear')
            
            # Run configuration
            if args.run_name:
                train_argv.extend(['--run_name', args.run_name])
            if args.base_dir:
                train_argv.extend(['--base_dir', args.base_dir])
            if args.root:
                train_argv.extend(['--root', args.root])
            
            # Device
            if args.device:
                train_argv.extend(['--device', args.device])
            if args.no_mixed_precision:
                train_argv.append('--no_mixed_precision')
            
            # Overlay options
            if args.no_overlay:
                train_argv.append('--no_overlay')
            elif args.overlay:
                train_argv.append('--overlay')
            if args.overlay_alpha:
                train_argv.extend(['--overlay_alpha', str(args.overlay_alpha)])
            if args.overlay_color:
                train_argv.extend(['--overlay_color', args.overlay_color])
            
            # Replace sys.argv and call train_main
            sys.argv = train_argv
            train_main()
        except ImportError as e:
            print(f"❌ Error: Could not import training module: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error during training: {e}")
            sys.exit(1)
        return
    
    setup_logging(verbose=(log_level == logging.DEBUG))
    
    # Resolve datasets
    try:
        datasets, modality = resolve_datasets(args.dataset, args.modality)
    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    if not datasets:
        logger.error(f"No datasets found for modality '{modality}'")
        sys.exit(1)
    
    logger.info(f"Processing {len(datasets)} dataset(s) from modality: {modality}")
    if len(datasets) > 1 and args.dataset is None:
        logger.info(f"Datasets: {', '.join(datasets[:5])}" + 
                   (f" ... and {len(datasets)-5} more" if len(datasets) > 5 else ""))
    
    # Process
    try:
        process_datasets(
            dataset_names=datasets,
            modality=modality,
            download=args.download,
            convert=args.convert,
            extract=args.extract,
            delete_raw_data=args.delete_raw_data,
            force_convert=args.force_convert,
            keep_raw_data=args.keep_raw_data,
            filter_empty=args.filter_empty_labels,
            check_integrity=args.check_integrity
        )
        logger.info("✅ All processing complete!")
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted by user")
        sys.exit(130)
    except PipelineError as e:
        logger.error(f"❌ {e}")
        for name, err in e.failures.items():
            logger.error(f"   - {name}: {type(err).__name__}: {err}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
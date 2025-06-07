"""
Visualization script for ImageCAS dataset using napari.

This script provides functionality to visualize:
1. Density images (*.img.nii.gz)
2. Label masks (*.label.nii.gz)
3. Separated artery masks (*.label.nii.gz from separated_arteries)

Usage:
    python -m src.visualization.vis_imagecas --case_id 1
    python -m src.visualization.vis_imagecas --img_file path/to/img.nii.gz --label_file path/to/label.nii.gz
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import napari
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
from src.utils.file_io import load_nii
from src.utils.logger import get_logger

def load_imagecas_data(
    case_id: Optional[int] = None,
    img_file: Optional[str] = None,
    label_file: Optional[str] = None,
    separated_label_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """Load ImageCAS data for visualization.
    
    Args:
        case_id: Case ID to load (e.g., 1 for 1.img.nii.gz)
        img_file: Direct path to image file
        label_file: Direct path to label file
        separated_label_file: Direct path to separated artery label file
        
    Returns:
        Tuple of (image_data, label_data, separated_label_data, nifti_info)
    """
    logger = get_logger()
    
    if case_id is not None:
        # Load from default paths
        data_root = Path("data/imageCAS")
        img_file = data_root / f"{case_id}.img.nii.gz"
        label_file = data_root / f"{case_id}.label.nii.gz"
        separated_label_file = Path("data/intermediate/separated_arteries") / f"{case_id}.label.nii.gz"
        
        if not img_file.exists():
            raise FileNotFoundError(f"Image file not found: {img_file}")
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
    
    # Load image data
    logger.info(f"Loading image from: {img_file}")
    image_data, image_nifti_info = load_nii(img_file)
    
    # Load label data
    logger.info(f"Loading label from: {label_file}")
    label_data, label_nifti_info = load_nii(label_file)
    label_data = label_data.astype(np.int32)  # Convert to integer type
    
    # Load separated label data if available
    separated_label_data = None
    separated_label_nifti_info = None
    if separated_label_file and os.path.exists(separated_label_file):
        logger.info(f"Loading separated label from: {separated_label_file}")
        separated_label_data, separated_label_nifti_info = load_nii(separated_label_file)
        separated_label_data = separated_label_data.astype(np.int32)  # Convert to integer type
    
    # Log volume information in a hierarchical format
    logger.info("=== Volume Information ===")
    
    # Image information
    logger.info("Image:")
    logger.info(f"  - Shape: {image_data.shape}")
    logger.info(f"  - Nifti Info:")
    logger.info(f"    * Shape: {image_nifti_info['shape']}")
    logger.info(f"    * Spacing: {image_nifti_info['spacing']}")
    logger.info(f"    * Origin: {image_nifti_info['origin']}")
    
    # Label information
    logger.info("Label:")
    logger.info(f"  - Shape: {label_data.shape}")
    logger.info(f"  - Nifti Info:")
    logger.info(f"    * Shape: {label_nifti_info['shape']}")
    logger.info(f"    * Spacing: {label_nifti_info['spacing']}")
    logger.info(f"    * Origin: {label_nifti_info['origin']}")
    
    # Separated label information (if available)
    if separated_label_data is not None:
        logger.info("Separated Label:")
        logger.info(f"  - Shape: {separated_label_data.shape}")
        logger.info(f"  - Nifti Info:")
        logger.info(f"    * Shape: {separated_label_nifti_info['shape']}")
        logger.info(f"    * Spacing: {separated_label_nifti_info['spacing']}")
        logger.info(f"    * Origin: {separated_label_nifti_info['origin']}")
    
    logger.info("========================")
    
    return image_data, label_data, separated_label_data, image_nifti_info

def visualize_imagecas(
    case_id: Optional[int] = None,
    img_file: Optional[str] = None,
    label_file: Optional[str] = None,
    separated_label_file: Optional[str] = None,
) -> None:
    """Visualize ImageCAS data using napari.
    
    Args:
        case_id: Case ID to load (e.g., 1 for 1.img.nii.gz)
        img_file: Direct path to image file
        label_file: Direct path to label file
        separated_label_file: Direct path to separated artery label file
    """
    logger = get_logger()
    
    # Load data
    image_data, label_data, separated_label_data, nifti_info = load_imagecas_data(
        case_id, img_file, label_file, separated_label_file
    )
    
    # Create napari viewer
    viewer = napari.Viewer(ndisplay=3)
    
    # Add image layer
    viewer.add_image(
        image_data,
        name="Density Image",
        colormap="gray",
        scale=nifti_info["spacing"],
        translate=nifti_info["origin"],
    )
    
    # Add label layer
    viewer.add_labels(
        label_data,
        name="Original Labels",
        scale=nifti_info["spacing"],
        translate=nifti_info["origin"],
    )
    
    # Add separated label layer if available
    if separated_label_data is not None:
        viewer.add_labels(
            separated_label_data,
            name="Separated Labels",
            scale=nifti_info["spacing"],
            translate=nifti_info["origin"],
        )
    
    # Set initial view
    # viewer.camera.zoom = 1.0
    # viewer.camera.center = np.array(image_data.shape) / 2
    
    # Start napari event loop
    napari.run()

def main():
    """Main function to parse arguments and start visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize ImageCAS data using napari"
    )
    
    # Add mutually exclusive group for input method
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--case_id",
        type=int,
        help="Case ID to visualize (e.g., 1 for 1.img.nii.gz)"
    )
    input_group.add_argument(
        "--img_file",
        type=str,
        help="Path to image file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--label_file",
        type=str,
        help="Path to label file (required if --img_file is used)"
    )
    parser.add_argument(
        "--separated_label_file",
        type=str,
        help="Path to separated artery label file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.img_file and not args.label_file:
        parser.error("--label_file is required when --img_file is used")
    
    # Start visualization
    visualize_imagecas(
        case_id=args.case_id,
        img_file=args.img_file,
        label_file=args.label_file,
        separated_label_file=args.separated_label_file,
    )

if __name__ == "__main__":
    main()

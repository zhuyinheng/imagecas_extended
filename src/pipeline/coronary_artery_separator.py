"""
Coronary artery separation and mesh generation module.

This module provides functionality to:
1. Separate coronary arteries into left and right branches
2. Generate surface meshes for each branch
3. Save the processed data in various formats (NIfTI, PLY, JSON)

The main processing steps are:
1. Load the original coronary artery mask and inlet annotations
2. Separate the arteries into left and right branches based on inlet positions
3. Generate surface meshes for each branch
4. Save the processed data with proper metadata

The output volume uses bitwise encoding for different labels:
- Bit 0 (1): Lumen mask
- Bit 1 (2): Left artery
- Bit 2 (4): Right artery
- Bit 3 (8): Left inlet
- Bit 4 (16): Right inlet
"""

import glob
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import rootutils
import sys
import logging
import argparse

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
from src.utils.file_io import (
    load_nii,
    save_nii,
    write_triangle_surface_mesh,
    write_json,
)
from src.utils.volume_utils import (
    connected_conponent_sorted_by_area,
    mask_mass_center,
    surface_from_mask_with_crop,
)
from src.utils.mesh_utils import fix_degenerate_mesh_np
from src.utils.logger import (
    get_logger,
    Timer,
    create_progress_bar,
    update_progress,
    start_progress,
    stop_progress,
)

# Constants for bit positions
LUMEN_BIT = 0
LEFT_ARTERY_BIT = 1
RIGHT_ARTERY_BIT = 2
LEFT_INLET_BIT = 3
RIGHT_INLET_BIT = 4

def decode_lumen_mask(encoded_volume: np.ndarray) -> np.ndarray:
    """Extract lumen mask (Bit 0) from encoded volume."""
    return (encoded_volume & (1 << LUMEN_BIT)) != 0

def decode_left_artery_mask(encoded_volume: np.ndarray) -> np.ndarray:
    """Extract left artery mask (Bit 1) from encoded volume."""
    return (encoded_volume & (1 << LEFT_ARTERY_BIT)) != 0

def decode_right_artery_mask(encoded_volume: np.ndarray) -> np.ndarray:
    """Extract right artery mask (Bit 2) from encoded volume."""
    return (encoded_volume & (1 << RIGHT_ARTERY_BIT)) != 0

def decode_left_inlet_mask(encoded_volume: np.ndarray) -> np.ndarray:
    """Extract left inlet mask (Bit 3) from encoded volume."""
    return (encoded_volume & (1 << LEFT_INLET_BIT)) != 0

def decode_right_inlet_mask(encoded_volume: np.ndarray) -> np.ndarray:
    """Extract right inlet mask (Bit 4) from encoded volume."""
    return (encoded_volume & (1 << RIGHT_INLET_BIT)) != 0

def decode_all_masks(encoded_volume: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract all masks from encoded volume into a dictionary."""
    return {
        'lumen': decode_lumen_mask(encoded_volume),
        'left_artery': decode_left_artery_mask(encoded_volume),
        'right_artery': decode_right_artery_mask(encoded_volume),
        'left_inlet': decode_left_inlet_mask(encoded_volume),
        'right_inlet': decode_right_inlet_mask(encoded_volume)
    }

def has_label(encoded_volume: np.ndarray, label_bits: int) -> np.ndarray:
    """Check if voxels have specific label bits set (e.g., 3 for bits 0 and 1)."""
    return (encoded_volume & label_bits) == label_bits

def separate_coronary_arteries(
    lumen_mask: np.ndarray,
    inlet_mask: np.ndarray,
) -> np.ndarray:
    """Separate coronary arteries into left and right branches.
    
    Args:
        lumen_mask: Binary mask of the coronary arteries
        inlet_mask: Mask containing left (1) and right (2) inlet points
        
    Returns:
        Multi-label mask with left (1) and right (2) arteries separated
    """
    logger = get_logger()
    
    with Timer("Finding left inlet position", logger=logger):
        left_inlet_center = mask_mass_center(inlet_mask == 1)
        logger.debug(f"Left inlet position: {left_inlet_center}")

    with Timer("Finding connected components", logger=logger):
        labeled_lumen_mask = connected_conponent_sorted_by_area(lumen_mask)
        component1_points = np.stack(np.where(labeled_lumen_mask == 1), -1)
        component2_points = np.stack(np.where(labeled_lumen_mask == 2), -1)
        logger.debug(f"Found {len(component1_points)} points in component 1, {len(component2_points)} points in component 2")
    
    with Timer("Calculating distances", logger=logger):
        dist_to_component1 = np.linalg.norm(
            component1_points - left_inlet_center[None, :], axis=-1
        ).min()
        dist_to_component2 = np.linalg.norm(
            component2_points - left_inlet_center[None, :], axis=-1
        ).min()
        logger.debug(f"Distance to component 1: {dist_to_component1:.2f}, to component 2: {dist_to_component2:.2f}")
    
    if dist_to_component1 > dist_to_component2:
        logger.debug("Swapping component labels")
        component1_mask = labeled_lumen_mask == 1
        component2_mask = labeled_lumen_mask == 2
        labeled_lumen_mask[component1_mask] = 2
        labeled_lumen_mask[component2_mask] = 1
        
    return labeled_lumen_mask


def process_single_case(
    imagecas_fn: str,
    inlet_fn: str,
    left_surface_fn: str,
    right_surface_fn: str,
    output_fn: str,
) -> None:
    """Process a single case from the ImageCAS dataset.
    
    Args:
        imagecas_fn: Path to the original ImageCAS mask
        inlet_fn: Path to the inlet annotation mask
        left_surface_fn: Path to save the left artery surface mesh
        right_surface_fn: Path to save the right artery surface mesh
        output_fn: Path to save the combined segmentation
        
    Note:
        The output mask uses bitwise encoding for different labels:
        - Bit 0 (1): Lumen mask
        - Bit 1 (2): Left artery
        - Bit 2 (4): Right artery
        - Bit 3 (8): Left inlet
        - Bit 4 (16): Right inlet
        This allows multiple labels to coexist at the same voxel.
    """
    logger = get_logger()
    logger.info(f"Processing {imagecas_fn}")

    try:
        with Timer("Loading data", logger=logger):
            lumen_mask, nifti_info = load_nii(imagecas_fn)
            lumen_mask = lumen_mask.astype(np.int32)
            inlet_mask, _ = load_nii(inlet_fn)
            original_dtype = inlet_mask.dtype
            inlet_mask = inlet_mask.astype(np.int32)
            logger.debug(f"Loaded masks with shapes: lumen {lumen_mask.shape}, inlet {inlet_mask.shape}")

        with Timer("Separating arteries", logger=logger):
            separated_mask = separate_coronary_arteries(lumen_mask, inlet_mask).astype(np.int32)
            logger.debug(f"Separated arteries into left and right branches")

        with Timer("Generating left surface mesh", logger=logger):
            left_vertices, left_faces, left_normals = surface_from_mask_with_crop(
                separated_mask == 1, nifti_info["spacing"], nifti_info["origin"]
            )
            left_vertices, left_faces, left_normals = fix_degenerate_mesh_np(
                left_vertices, left_faces, left_normals
            )
            write_triangle_surface_mesh(left_surface_fn, left_vertices, left_faces, left_normals)
            logger.debug(f"Generated left surface mesh with {len(left_vertices)} vertices, {len(left_faces)} faces")

        with Timer("Generating right surface mesh", logger=logger):
            right_vertices, right_faces, right_normals = surface_from_mask_with_crop(
                separated_mask == 2, nifti_info["spacing"], nifti_info["origin"]
            )
            right_vertices, right_faces, right_normals = fix_degenerate_mesh_np(
                right_vertices, right_faces, right_normals
            )
            write_triangle_surface_mesh(right_surface_fn, right_vertices, right_faces, right_normals)
            logger.debug(f"Generated right surface mesh with {len(right_vertices)} vertices, {len(right_faces)} faces")

        with Timer("Combining segmentations", logger=logger):
            # Initialize with zeros
            combined_mask = np.zeros_like(lumen_mask).astype(np.uint8)
            
            # Set individual bits for each label
            # Bit 0 (1): Lumen mask
            combined_mask |= (lumen_mask == 1).astype(np.uint8)
            
            # Bit 1 (2): Left artery
            combined_mask |= ((separated_mask == 1).astype(np.uint8) << 1)
            
            # Bit 2 (4): Right artery
            combined_mask |= ((separated_mask == 2).astype(np.uint8) << 2)
            
            # Bit 3 (8): Left inlet
            combined_mask |= ((inlet_mask == 1).astype(np.uint8) << 3)
            
            # Bit 4 (16): Right inlet
            combined_mask |= ((inlet_mask == 2).astype(np.uint8) << 4)

            save_nii(
                combined_mask.astype(original_dtype),
                output_fn,
                ref_nii_path=imagecas_fn,
            )
            logger.debug("Saved combined segmentation")

        with Timer("Saving inlet positions", logger=logger):
            left_inlet_pos = np.stack(np.where(inlet_mask == 1), -1)[0]  # volume index
            right_inlet_pos = np.stack(np.where(inlet_mask == 2), -1)[0]  # volume index
            left_inlet_pos = left_inlet_pos * nifti_info["spacing"] + nifti_info["origin"]
            right_inlet_pos = right_inlet_pos * nifti_info["spacing"] + nifti_info["origin"]
            write_json(left_surface_fn + ".inlet.json", left_inlet_pos.tolist())
            write_json(right_surface_fn + ".inlet.json", right_inlet_pos.tolist())
            logger.debug(f"Saved inlet positions: left {left_inlet_pos}, right {right_inlet_pos}")
        
        logger.info(f"Successfully processed {imagecas_fn}")
    except Exception as e:
        error_msg = f"Error processing {imagecas_fn}:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise

def main():
    """Main processing function."""
    label_encoding = """
    The output mask uses bitwise encoding for different labels:
    - Bit 0 (1): Lumen mask
    - Bit 1 (2): Left artery
    - Bit 2 (4): Right artery
    - Bit 3 (8): Left inlet
    - Bit 4 (16): Right inlet
    
    This encoding allows multiple labels to coexist at the same voxel.
    For example:
    - Value 3 (binary 11) means both lumen and left artery
    - Value 5 (binary 101) means both lumen and right artery
    - Value 9 (binary 1001) means both lumen and left inlet
    """

    parser = argparse.ArgumentParser(
        description=(
            "Separate coronary arteries into left and right branches, "
            "generate surface meshes, and save processed data"
        )
    )
    parser.add_argument(
        "--imagecas_root",
        type=str,
        default="data/imageCAS",
        help="Root path of the ImageCAS dataset containing *.label.nii.gz files"
    )
    parser.add_argument(
        "--annotation_root",
        type=str,
        default="data/intermediate/inlet_annotation",
        help="Root path of the inlet annotation dataset containing *.label.nii.gz files"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/intermediate/separated_arteries",
        help=(
            "Root path for output files. Will contain:\n"
            "1. *.label.nii.gz files with bitwise encoded labels:\n"
            f"{label_encoding}\n"
            "2. *.left.ply and *.right.ply surface meshes\n"
            "3. *.inlet.json files with inlet positions"
        )
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode with only 3 cases"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )

    args = parser.parse_args()

    # Set up logger
    logger = get_logger(
        "coronary_artery_separator",
        "logs",
        level=logging.DEBUG if args.debug else logging.INFO,
        debug=args.debug
    )

    try:
        # Create output directory
        os.makedirs(args.output_root, exist_ok=True)

        # Prepare file lists
        with Timer("Finding input files", logger=logger):
            imagecas_files = sorted(glob.glob(os.path.join(args.imagecas_root, "*.label.nii.gz")))
            if args.test_mode:
                imagecas_files = imagecas_files[:3]
                logger.info("Running in test mode with 3 cases")
            logger.info(f"Found {len(imagecas_files)} cases to process")
            logger.debug(f"Found files: {imagecas_files}")

        # Start progress display
        start_progress()
        progress_task_id = create_progress_bar(len(imagecas_files), "Processing cases")

        for imagecas_file in imagecas_files:
            case_id = os.path.basename(imagecas_file).split(".")[0]
            inlet_file = os.path.join(args.annotation_root, f"{case_id}.label.nii.gz")
            left_surface_file = os.path.join(args.output_root, f"{case_id}.left.ply")
            right_surface_file = os.path.join(args.output_root, f"{case_id}.right.ply")
            output_file = os.path.join(args.output_root, f"{case_id}.label.nii.gz")

            process_single_case(
                imagecas_fn=imagecas_file,
                inlet_fn=inlet_file,
                left_surface_fn=left_surface_file,
                right_surface_fn=right_surface_file,
                output_fn=output_file,
            )
            update_progress(progress_task_id)

        # Stop progress display
        stop_progress()

    except Exception as e:
        error_msg = f"Fatal error in main process:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()

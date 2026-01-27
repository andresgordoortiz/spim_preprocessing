#!/usr/bin/env python3
"""
SPIM Preprocessing Script with Metadata Preservation
Adapted for Nextflow pipeline integration
"""

import argparse
import psutil
import time
import os
import sys
import json
import numpy as np
import tifffile
import scipy.ndimage as ndi
from skimage.transform import rescale, resize
import cv2
import pims
from WBNS import WBNS_image
import RedLionfishDeconv as rl
from tqdm import tqdm
from typing import Optional, Tuple
import subprocess

# Import all functions from original script
from spim_pipeline_fixed import (
    image_scaling_intens,
    read_tiff_voxel_size,
    read_nd2_voxel_size,
    z_intensity_correction,
    shading_correct_xy_estimated,
    clahe_3d_stack,
    reslice,
    image_postprocessing,
    getNormalizationThresholds,
    remove_outliers_image,
    print_resource_usage
)

def preserve_metadata_in_tiff(
    output_path: str,
    image_data: np.ndarray,
    x_resolution_um: float,
    y_resolution_um: float,
    z_spacing_um: float,
    timepoint: int = 0,
    additional_metadata: dict = None
):
    """
    Save TIFF with complete metadata preservation for TrackMate compatibility.

    Parameters:
    -----------
    output_path : str
        Output file path
    image_data : np.ndarray
        Image data (ZYX or TZYX)
    x_resolution_um : float
        X pixel size in microns
    y_resolution_um : float
        Y pixel size in microns
    z_spacing_um : float
        Z spacing in microns
    timepoint : int
        Timepoint index
    additional_metadata : dict
        Additional metadata to include
    """

    # Prepare ImageJ-compatible metadata
    imagej_metadata = {
        'spacing': z_spacing_um,
        'unit': 'um',
        'TimePoint': timepoint
    }

    # Determine axes based on dimensions
    if image_data.ndim == 3:
        imagej_metadata['axes'] = 'ZYX'
        imagej_metadata['slices'] = image_data.shape[0]
    elif image_data.ndim == 4:
        imagej_metadata['axes'] = 'TZYX'
        imagej_metadata['frames'] = image_data.shape[0]
        imagej_metadata['slices'] = image_data.shape[1]

    # Add any additional metadata
    if additional_metadata:
        imagej_metadata.update(additional_metadata)

    # Save with full metadata
    tifffile.imwrite(
        output_path,
        image_data.astype(np.uint16),
        imagej=True,
        resolution=(1.0/x_resolution_um, 1.0/y_resolution_um),
        metadata=imagej_metadata
    )

    print(f"  Saved with metadata: X={x_resolution_um:.4f} um, "
          f"Y={y_resolution_um:.4f} um, Z={z_spacing_um:.4f} um")

def main():
    parser = argparse.ArgumentParser(
        description="SPIM Image Preprocessing Pipeline with Metadata Preservation"
    )

    # Paths
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--outdir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--psf_path", type=str, required=True,
                       help="Path to PSF model")
    parser.add_argument("--metadata_json", type=str,
                       help="Path to metadata JSON (optional, will extract if not provided)")

    # Image Parameters
    parser.add_argument("--image_scaling", type=float, default=1.0,
                       help="Image scaling factor")
    parser.add_argument("--xy_pixel", type=float, default=0.0,
                       help="Force XY pixel size (um). 0 to read from metadata")
    parser.add_argument("--z_pixel", type=float, default=0.0,
                       help="Force Z pixel size (um). 0 to read from metadata")

    # Processing Flags
    parser.add_argument("--no_clahe", action="store_true",
                       help="Disable CLAHE")
    parser.add_argument("--no_z_correction", action="store_true",
                       help="Disable Z intensity correction")
    parser.add_argument("--no_shading", action="store_true",
                       help="Disable Shading correction")

    # Deconvolution Params
    parser.add_argument("--padding", type=int, default=32,
                       help="Padding for deconvolution")
    parser.add_argument("--niter", type=int, default=3,
                       help="Iterations for 3D Deconvolution")
    parser.add_argument("--niterz", type=int, default=3,
                       help="Iterations for 2D XZ Deconvolution")

    # Normalization Params
    parser.add_argument("--min_v", type=float, default=0,
                       help="Min value for normalization")
    parser.add_argument("--max_v", type=float, default=65535,
                       help="Max value for normalization")
    parser.add_argument("--percentile_low", type=float, default=40,
                       help="Low percentile for outlier removal")
    parser.add_argument("--percentile_high", type=float, default=99.99,
                       help="High percentile for outlier removal")

    # Background / Post-processing
    parser.add_argument("--resolution_px0", type=float, default=10,
                       help="BG Subtraction resolution")
    parser.add_argument("--resolution_pz0", type=float, default=10,
                       help="BG Subtraction resolution Z")
    parser.add_argument("--noise_lvl", type=int, default=2,
                       help="Noise level (MUST BE INTEGER)")
    parser.add_argument("--sigma", type=float, default=1.0,
                       help="Gaussian smoothing sigma")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file does not exist: {args.input_file}")
        sys.exit(1)

    if args.niter > 0 or args.niterz > 0:
        if not os.path.isfile(args.psf_path):
            print(f"ERROR: PSF file does not exist: {args.psf_path}")
            sys.exit(1)

    # Log parameters
    print("\n" + "=" * 60)
    print("SPIM PREPROCESSING WITH METADATA PRESERVATION")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 60 + "\n")

    # Mapping boolean flags
    apply_clahe = not args.no_clahe
    apply_z_intensity_correction = not args.no_z_correction
    apply_shading_correct = not args.no_shading
    percentiles_source = (args.percentile_low, args.percentile_high)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    # PSF Loading
    if args.niter > 0:
        t0 = time.time()
        print(f"Loading PSF from {args.psf_path}")
        psf = tifffile.imread(args.psf_path)
        psf_shape = psf.shape
        if args.image_scaling > 0 and args.image_scaling != 1.0:
            psf = rescale(
                psf,
                (args.image_scaling, args.image_scaling, args.image_scaling),
                order=3,
                preserve_range=True,
                anti_aliasing=True
            )
            print(f"     -PSF dimension from : {psf_shape} to {psf.shape}")
        psf_f = psf.astype(np.float32)
        psf = psf_f / psf_f.sum()
        print(f"[Timer] PSF preparation took {time.time() - t0:.2f} seconds")

    # Load metadata if provided
    original_metadata = None
    if args.metadata_json and os.path.exists(args.metadata_json):
        with open(args.metadata_json, 'r') as f:
            original_metadata = json.load(f)
        print(f"Loaded metadata from: {args.metadata_json}")

    # Processing Single Image
    image_path = args.input_file
    image_name = os.path.basename(image_path)

    print(f"\n[Processing] {image_name}")
    print_resource_usage()

    start_time_total = time.time()

    # Load Image
    t0 = time.time()
    ext = os.path.splitext(image_name)[1].lower()

    try:
        if ext in ['.tif', '.tiff']:
            print("  Loading TIFF image...")
            img = tifffile.imread(image_path).astype(np.uint16)
            voxel_size = read_tiff_voxel_size(image_path)
        elif ext == '.nd2':
            print("  Loading ND2 image...")
            img = pims.open(image_path)
            voxel_size = read_nd2_voxel_size(img)
            img = np.array(img, dtype=np.uint16, copy=False)
        else:
            print(f"ERROR: Unsupported format: {ext}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        sys.exit(1)

    t1 = time.time()
    print(f"[Timer] Image loading took {t1 - t0:.2f} seconds")
    print(f"  - shape: {img.shape}, dtype: {img.dtype}")
    print(f"  - estimated size (GB): {img.nbytes / (1024**3):.3f}")
    print_resource_usage()

    physical_pixel_sizeX, physical_pixel_sizeY, physical_pixel_sizeZ = voxel_size

    # Override with metadata if available
    if original_metadata:
        physical_pixel_sizeX = original_metadata.get('x_resolution_um', physical_pixel_sizeX)
        physical_pixel_sizeY = original_metadata.get('y_resolution_um', physical_pixel_sizeY)
        if 'imagej' in original_metadata:
            physical_pixel_sizeZ = original_metadata['imagej'].get('spacing', physical_pixel_sizeZ)

    # Override with manual values if provided
    if args.xy_pixel > 0:
        physical_pixel_sizeX = args.xy_pixel
    if args.z_pixel > 0:
        physical_pixel_sizeZ = args.z_pixel

    print(f"  - voxel sizes (um): X={physical_pixel_sizeX:.4f}, "
          f"Y={physical_pixel_sizeY:.4f}, Z={physical_pixel_sizeZ:.4f}")

    # Store original dimensions for metadata tracking
    original_shape = img.shape
    original_x_res = physical_pixel_sizeX
    original_y_res = physical_pixel_sizeY
    original_z_res = physical_pixel_sizeZ

    # Image scaling (XY only)
    if args.image_scaling > 0 and args.image_scaling != 1.0:
        t0 = time.time()
        img_shape = img.shape
        print(f"  - image dimension : {img.shape}, scaling {args.image_scaling}")
        img = rescale(
            img,
            (1.0, args.image_scaling, args.image_scaling),
            order=3,
            preserve_range=True,
            anti_aliasing=True
        )
        physical_pixel_sizeX /= args.image_scaling
        physical_pixel_sizeY /= args.image_scaling
        print(f"  - image dimension from : {img_shape} to {img.shape}")
        t1 = time.time()
        print(f"[Timer] Image rescaling took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Store shape before reslicing
    img_shape = img.shape
    scale = physical_pixel_sizeX / physical_pixel_sizeZ

    # Pre-processing
    if apply_shading_correct:
        t0 = time.time()
        print("[Check-in] Running shading_correct_xy_estimated...")
        img, field = shading_correct_xy_estimated(
            img, sigma_xy=96, z_axis=0, per_slice=False
        )
        t1 = time.time()
        print(f"[Timer] Shading correction took {t1 - t0:.2f} seconds")
        print_resource_usage()

    if apply_z_intensity_correction:
        t0 = time.time()
        print("[Check-in] Running z_intensity_correction...")
        img, scales = z_intensity_correction(
            img, z_axis=0, method="p95", smooth_window=11
        )
        t1 = time.time()
        print(f"[Timer] Z-intensity correction took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Isotropic Reslicing
    if abs(1.0 - scale) > 1e-4:
        t0 = time.time()
        print("[Check-in] Reslicing to isotropic...")
        img = reslice(img, 'xy', physical_pixel_sizeX, physical_pixel_sizeZ)
        t1 = time.time()
        print(f"[Timer] Reslicing took {t1 - t0:.2f} seconds")

    img = img.astype(np.float32)
    new_img_shape = img.shape

    # Recalculate voxel size after reslicing
    new_physical_pixel_sizeZ = (
        img_shape[0] * physical_pixel_sizeZ / new_img_shape[0]
    )
    print(f"  - image dimension from : {img_shape} to {new_img_shape} "
          "after isotropic interpolation")
    print(f"  - z-space from : {physical_pixel_sizeZ:.4f} to "
          f"{new_physical_pixel_sizeZ:.4f}")
    physical_pixel_sizeZ = new_physical_pixel_sizeZ
    print_resource_usage()

    # Recalculate resolution for BG subtraction
    resolution_px = int(args.resolution_px0 / new_physical_pixel_sizeZ)
    resolution_pz = int(args.resolution_pz0 / new_physical_pixel_sizeZ)
    print(f"  BG subtraction : {resolution_px},  {resolution_pz}")

    # Deconvolution (GPU)
    if args.niter > 0:
        t0 = time.time()
        print("[Check-in] Running 3D deconvolution...")
        img = image_scaling_intens(img, args.min_v, args.max_v, True)
        img = np.pad(img, args.padding, mode='reflect')
        imgSizeGB = img.nbytes / (1024 ** 3)
        print(f'    -size(GB) : {imgSizeGB:.3f}')
        print_resource_usage()
        res_gpu = rl.doRLDeconvolutionFromNpArrays(
            img, psf, niter=args.niter, resAsUint8=False
        )
        img = res_gpu[
            args.padding:-args.padding,
            args.padding:-args.padding,
            args.padding:-args.padding
        ]
        t1 = time.time()
        print(f"[Timer] 3D deconvolution took {t1 - t0:.2f} seconds")
        print_resource_usage()

    if args.niterz > 0:
        t0 = time.time()
        print("[Check-in] Running 2D (XZ) deconvolution...")
        img = image_scaling_intens(img, args.min_v, args.max_v, True)
        img_xz = np.transpose(img, [1, 0, 2])
        psf_xz = np.transpose(psf, [1, 0, 2])
        img_xz = np.pad(img_xz, args.padding, mode='reflect')
        imgSizeGB = img_xz.nbytes / (1024 ** 3)
        print(f'    img_xz -size(GB) : {imgSizeGB:.3f}')
        print_resource_usage()
        res_gpu = rl.doRLDeconvolutionFromNpArrays(
            img_xz, psf_xz, niter=args.niter, resAsUint8=False
        )
        img_xz = res_gpu[
            args.padding:-args.padding,
            args.padding:-args.padding,
            args.padding:-args.padding
        ]
        img = np.transpose(img_xz, [1, 0, 2])
        t1 = time.time()
        print(f"[Timer] 2D (XZ) deconvolution took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Post-processing
    t0 = time.time()
    print("[Check-in] Running post-processing...")
    img = image_postprocessing(
        img, resolution_px, resolution_pz, args.noise_lvl, args.sigma
    )
    t1 = time.time()
    print(f"[Timer] Post-processing took {t1 - t0:.2f} seconds")
    print_resource_usage()

    if apply_clahe:
        t0 = time.time()
        print("[Check-in] Applying CLAHE...")
        img_xz = np.transpose(img, [1, 0, 2])
        img_xz = clahe_3d_stack(
            img_xz, clip_limit=0.01, kernel_size=(64, 64), axis=0
        )
        img = np.transpose(img_xz, [1, 0, 2])
        t1 = time.time()
        print(f"[Timer] CLAHE took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Normalization
    if percentiles_source[0] > 0 or percentiles_source[1] < 100:
        t0 = time.time()
        print("[Check-in] Removing outliers and normalizing intensities...")
        low_thres, high_thres = getNormalizationThresholds(img, percentiles_source)
        img = remove_outliers_image(img, low_thres, high_thres)
        t1 = time.time()
        print(f"[Timer] Outlier removal and normalization took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Final intensity scaling
    t0 = time.time()
    print("[Check-in] Final intensity scaling and saving...")
    img = image_scaling_intens(img, args.min_v, args.max_v, True)
    img = img.astype(np.uint16)
    t1 = time.time()
    print(f"[Timer] Final scaling and conversion took {t1 - t0:.2f} seconds")

    # Save with metadata preservation
    t0 = time.time()
    base_name = os.path.splitext(image_name)[0]
    image_out_name = f"{base_name}_processed.tif"
    img_out_path = os.path.join(args.outdir, image_out_name)

    # Extract timepoint if present in original metadata
    timepoint = 0
    if original_metadata and 'imagej' in original_metadata:
        timepoint = original_metadata['imagej'].get('TimePoint', 0)

    # Save with full metadata
    preserve_metadata_in_tiff(
        img_out_path,
        img,
        physical_pixel_sizeX,
        physical_pixel_sizeY,
        physical_pixel_sizeZ,
        timepoint=timepoint,
        additional_metadata={
            'ProcessedBy': 'SPIM_Pipeline_v1.0',
            'ProcessingDate': time.strftime('%Y-%m-%d %H:%M:%S'),
            'OriginalShape': list(original_shape),
            'ScalingFactor': args.image_scaling
        }
    )

    t1 = time.time()
    print(f"  Saved processed image to: {img_out_path}")
    print(f"[Timer] Saving image took {t1 - t0:.2f} seconds")

    # Validate output
    if not os.path.isfile(img_out_path):
        print(f"ERROR: Output file was not created: {img_out_path}")
        sys.exit(1)

    output_size = os.path.getsize(img_out_path)
    if output_size == 0:
        print(f"ERROR: Output file is empty: {img_out_path}")
        sys.exit(1)

    print(f"[Success] Output size: {output_size / (1024**2):.2f} MB")
    elapsed_time = time.time() - start_time_total
    print(f"[Done] Elapsed Time: {elapsed_time:.4f} seconds")
    print_resource_usage()

if __name__ == "__main__":
    main()
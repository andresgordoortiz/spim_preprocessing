#!/usr/bin/env nextflow

/*
 * ============================================================================
 * SPIM 4D Image Processing Pipeline
 * ============================================================================
 * 
 * This pipeline processes 4D SPIM images through:
 * 1. Preprocessing and deconvolution
 * 2. Cellpose segmentation
 * 3. TrackMate tracking
 * 
 * All parameters are loaded from a JSON configuration file for reproducibility.
 * Metadata is preserved throughout the pipeline for TrackMate compatibility.
 */

nextflow.enable.dsl=2

// ============================================================================
// PARAMETER DEFINITIONS
// ============================================================================

params.input_dir = null
params.output_dir = null
params.config_json = null
params.container = "docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-8720eea"
params.help = false

// Show help message
if (params.help) {
    log.info """
    ============================================================================
    SPIM 4D Image Processing Pipeline
    ============================================================================
    
    Usage:
        nextflow run spim_pipeline.nf \\
            --input_dir <path> \\
            --output_dir <path> \\
            --config_json <path> \\
            [options]
    
    Required Arguments:
        --input_dir         Directory containing input 4D TIFF images
        --output_dir        Directory for all pipeline outputs
        --config_json       JSON file with processing parameters
    
    Optional Arguments:
        --container         Singularity/Docker container image
                           (default: docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-8720eea)
        --help             Show this help message
    
    Output Structure:
        output_dir/
        ├── 01_preprocessed/     - Preprocessed & deconvolved images
        ├── 02_segmented/        - Cellpose segmentation masks
        ├── 03_tracked/          - TrackMate tracking results
        ├── metadata/            - Preserved metadata files
        ├── logs/                - Processing logs
        └── reports/             - QC reports and visualizations
    
    Example:
        nextflow run spim_pipeline.nf \\
            --input_dir /data/raw_images \\
            --output_dir /data/processed \\
            --config_json config.json
    
    ============================================================================
    """.stripIndent()
    exit 0
}

// Validate required parameters
if (!params.input_dir || !params.output_dir || !params.config_json) {
    log.error "ERROR: Missing required parameters!"
    log.error "Required: --input_dir, --output_dir, --config_json"
    log.error "Run with --help for usage information"
    exit 1
}

// ============================================================================
// LOAD CONFIGURATION
// ============================================================================

def loadConfig(json_path) {
    def jsonSlurper = new groovy.json.JsonSlurper()
    return jsonSlurper.parse(new File(json_path))
}

config = loadConfig(params.config_json)

// ============================================================================
// WORKFLOW HEADER
// ============================================================================

log.info """
============================================================================
SPIM 4D Image Processing Pipeline
============================================================================
Input directory    : ${params.input_dir}
Output directory   : ${params.output_dir}
Configuration      : ${params.config_json}
Container          : ${params.container}

Pipeline steps:
  1. Preprocessing & Deconvolution
  2. Cellpose Segmentation
  3. TrackMate Tracking

Started at: ${new Date()}
============================================================================
""".stripIndent()

// ============================================================================
// PROCESS: Extract and Preserve Metadata
// ============================================================================

process EXTRACT_METADATA {
    tag "${image_file.baseName}"
    
    publishDir "${params.output_dir}/metadata", 
        mode: 'copy',
        pattern: "*.json"
    
    input:
    path image_file
    
    output:
    tuple path(image_file), path("${image_file.baseName}_metadata.json"), emit: with_metadata
    path "${image_file.baseName}_metadata.json", emit: metadata_only
    
    script:
    """
    #!/usr/bin/env python3
    import json
    import tifffile
    import numpy as np
    from pathlib import Path
    
    # Read TIFF metadata
    with tifffile.TiffFile('${image_file}') as tif:
        metadata = {}
        
        # ImageJ metadata
        if tif.imagej_metadata:
            metadata['imagej'] = {
                'spacing': tif.imagej_metadata.get('spacing', 1.0),
                'unit': tif.imagej_metadata.get('unit', 'micron'),
                'axes': tif.imagej_metadata.get('axes', 'TZYX'),
                'frames': tif.imagej_metadata.get('frames', 1),
                'slices': tif.imagej_metadata.get('slices', 1),
                'channels': tif.imagej_metadata.get('channels', 1),
            }
        
        # TIFF tags from first page
        first_page = tif.pages[0]
        tags = first_page.tags
        
        # Extract resolution
        if 'XResolution' in tags:
            x_num, x_denom = tags['XResolution'].value
            metadata['x_resolution_um'] = x_denom / x_num
        else:
            metadata['x_resolution_um'] = 1.0
            
        if 'YResolution' in tags:
            y_num, y_denom = tags['YResolution'].value
            metadata['y_resolution_um'] = y_denom / y_num
        else:
            metadata['y_resolution_um'] = 1.0
        
        # Image dimensions
        metadata['shape'] = {
            'axes': 'TZYX' if len(tif.series[0].shape) == 4 else 'ZYX',
            'dimensions': list(tif.series[0].shape)
        }
        
        # Data type
        metadata['dtype'] = str(tif.series[0].dtype)
        
        # Software info
        if 'Software' in tags:
            metadata['software'] = tags['Software'].value
    
    # Save metadata
    with open('${image_file.baseName}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Extracted metadata for ${image_file.baseName}")
    print(json.dumps(metadata, indent=2))
    """
}

// ============================================================================
// PROCESS: Split 4D Image into Timepoints
// ============================================================================

process SPLIT_TIMEPOINTS {
    tag "${image_file.baseName}"
    
    publishDir "${params.output_dir}/01_preprocessed/timepoints", 
        mode: 'copy',
        pattern: "*_t*.tif"
    
    input:
    tuple path(image_file), path(metadata_json)
    
    output:
    tuple val("${image_file.baseName}"), 
          path("${image_file.baseName}_t*.tif"), 
          path(metadata_json), emit: timepoints
    
    script:
    """
    #!/usr/bin/env python3
    import tifffile
    import numpy as np
    import json
    from pathlib import Path
    
    print("Loading image: ${image_file}")
    img = tifffile.imread('${image_file}')
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    
    # Load metadata
    with open('${metadata_json}', 'r') as f:
        metadata = json.load(f)
    
    # Determine if 4D (TZYX) or 3D (ZYX)
    if img.ndim == 4:
        print(f"Processing 4D image with {img.shape[0]} timepoints")
        n_timepoints = img.shape[0]
        
        for t in range(n_timepoints):
            timepoint_img = img[t]
            output_name = f"${image_file.baseName}_t{t:04d}.tif"
            
            # Save with metadata preservation
            tifffile.imwrite(
                output_name,
                timepoint_img,
                imagej=True,
                resolution=(1.0/metadata['x_resolution_um'], 
                           1.0/metadata['y_resolution_um']),
                metadata={
                    'spacing': metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0,
                    'unit': 'um',
                    'axes': 'ZYX',
                    'TimePoint': t
                }
            )
            print(f"  Saved timepoint {t}: {output_name}")
    
    elif img.ndim == 3:
        print("Processing 3D image (single timepoint)")
        output_name = f"${image_file.baseName}_t0000.tif"
        
        tifffile.imwrite(
            output_name,
            img,
            imagej=True,
            resolution=(1.0/metadata['x_resolution_um'], 
                       1.0/metadata['y_resolution_um']),
            metadata={
                'spacing': metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0,
                'unit': 'um',
                'axes': 'ZYX',
                'TimePoint': 0
            }
        )
        print(f"  Saved single timepoint: {output_name}")
    
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}D")
    
    print("Timepoint splitting completed")
    """
}

// ============================================================================
// PROCESS: Preprocess and Deconvolve Single Timepoint
// ============================================================================

process PREPROCESS_DECONVOLVE {
    tag "${series_name}_${timepoint_file.baseName}"
    
    publishDir "${params.output_dir}/01_preprocessed/processed", 
        mode: 'copy',
        pattern: "*_processed.tif"
    
    publishDir "${params.output_dir}/logs/preprocessing",
        mode: 'copy',
        pattern: "*.log"
    
    container params.container
    
    input:
    tuple val(series_name), 
          path(timepoint_file), 
          path(metadata_json)
    val preprocess_config
    
    output:
    tuple val(series_name),
          path("${timepoint_file.baseName}_processed.tif"),
          path(metadata_json), emit: processed
    path "${timepoint_file.baseName}_preprocess.log", emit: log
    
    script:
    def cfg = preprocess_config
    """
    #!/bin/bash
    set -euo pipefail
    
    echo "============================================"
    echo "Preprocessing: ${timepoint_file.baseName}"
    echo "============================================"
    
    # Run preprocessing with all parameters from config
    python3 << 'PYTHON_EOF'
import json
import sys
import os

# Load metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load config
config = ${groovy.json.JsonOutput.toJson(cfg)}

# Build command
cmd = [
    'python', 'spim_pipeline_fixed.py',
    '--input_file', '${timepoint_file}',
    '--outdir', '.',
    '--psf_path', config['psf_path'],
    '--image_scaling', str(config['image_scaling']),
    '--niter', str(config['niter']),
    '--niterz', str(config['niterz']),
    '--percentile_low', str(config['percentile_low']),
    '--percentile_high', str(config['percentile_high']),
    '--sigma', str(config['sigma']),
    '--min_v', str(config['min_v']),
    '--max_v', str(config['max_v']),
    '--resolution_px0', str(config['resolution_px0']),
    '--resolution_pz0', str(config['resolution_pz0']),
    '--noise_lvl', str(config['noise_lvl']),
    '--padding', str(config['padding'])
]

# Add optional flags
if config.get('no_clahe', False):
    cmd.append('--no_clahe')
if config.get('no_z_correction', False):
    cmd.append('--no_z_correction')
if config.get('no_shading', False):
    cmd.append('--no_shading')

print("Command:", ' '.join(cmd))
print("\\n" + "="*50)

# Execute
import subprocess
result = subprocess.run(cmd, capture_output=True, text=True)

# Save log
with open('${timepoint_file.baseName}_preprocess.log', 'w') as f:
    f.write("STDOUT:\\n")
    f.write(result.stdout)
    f.write("\\n\\nSTDERR:\\n")
    f.write(result.stderr)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)

sys.exit(result.returncode)
PYTHON_EOF

    # Rename output to standard format
    ORIGINAL_OUTPUT=\$(ls *_${cfg.image_scaling.toString().replace('.', '')}*.tif 2>/dev/null | head -1)
    if [ -n "\$ORIGINAL_OUTPUT" ]; then
        mv "\$ORIGINAL_OUTPUT" "${timepoint_file.baseName}_processed.tif"
        echo "Renamed output to: ${timepoint_file.baseName}_processed.tif"
    else
        echo "ERROR: No processed output found"
        exit 1
    fi
    
    # Restore metadata to processed image
    python3 << 'RESTORE_META'
import tifffile
import json

# Load original metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load processed image
img = tifffile.imread('${timepoint_file.baseName}_processed.tif')

# Recalculate voxel sizes after scaling
x_res = metadata['x_resolution_um'] / ${cfg.image_scaling}
y_res = metadata['y_resolution_um'] / ${cfg.image_scaling}
z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

# Re-save with preserved metadata
tifffile.imwrite(
    '${timepoint_file.baseName}_processed.tif',
    img,
    imagej=True,
    resolution=(1.0/x_res, 1.0/y_res),
    metadata={
        'spacing': z_spacing,
        'unit': 'um',
        'axes': 'ZYX',
        'TimePoint': metadata['imagej'].get('TimePoint', 0) if 'imagej' in metadata else 0
    }
)

print(f"Metadata restored: X={x_res:.4f} um, Y={y_res:.4f} um, Z={z_spacing:.4f} um")
RESTORE_META
    
    echo "Preprocessing completed: ${timepoint_file.baseName}_processed.tif"
    """
}

// ============================================================================
// PROCESS: Cellpose Segmentation
// ============================================================================

process CELLPOSE_SEGMENT {
    tag "${series_name}_${processed_file.baseName}"
    
    publishDir "${params.output_dir}/02_segmented", 
        mode: 'copy',
        pattern: "*_Cellseg.tif"
    
    publishDir "${params.output_dir}/logs/segmentation",
        mode: 'copy',
        pattern: "*.log"
    
    container params.container
    
    input:
    tuple val(series_name),
          path(processed_file),
          path(metadata_json)
    val segment_config
    
    output:
    tuple val(series_name),
          path("${processed_file.baseName}_Cellseg.tif"),
          path(metadata_json), emit: segmented
    path "${processed_file.baseName}_segment.log", emit: log
    
    script:
    def cfg = segment_config
    """
    #!/bin/bash
    set -euo pipefail
    
    echo "============================================"
    echo "Segmentation: ${processed_file.baseName}"
    echo "============================================"
    
    # Build Cellpose command
    CELLPOSE_CMD="cellpose --image_path ${processed_file}"
    CELLPOSE_CMD+=" --savedir ."
    CELLPOSE_CMD+=" --pretrained_model ${cfg.model}"
    CELLPOSE_CMD+=" --diameter ${cfg.diameter}"
    CELLPOSE_CMD+=" --flow_threshold ${cfg.flow_threshold}"
    CELLPOSE_CMD+=" --cellprob_threshold ${cfg.cellprob_threshold}"
    
    ${cfg.use_gpu ? 'CELLPOSE_CMD+=" --use_gpu"' : ''}
    ${cfg.do_3d ? 'CELLPOSE_CMD+=" --do_3D"' : ''}
    ${cfg.save_tif ? 'CELLPOSE_CMD+=" --save_tif"' : ''}
    ${cfg.save_flows ? 'CELLPOSE_CMD+=" --save_flows"' : ''}
    ${cfg.save_npy ? '' : 'CELLPOSE_CMD+=" --no_npy"'}
    
    CELLPOSE_CMD+=" --verbose"
    
    echo "Running: \$CELLPOSE_CMD"
    echo ""
    
    # Run Cellpose and capture output
    \$CELLPOSE_CMD 2>&1 | tee ${processed_file.baseName}_segment.log
    
    # Rename output to include diameter (matching ImageJ convention)
    CELLPOSE_OUTPUT="${processed_file.baseName}_cp_masks.tif"
    FINAL_OUTPUT="${processed_file.baseName}_diam${cfg.diameter}_Cellseg.tif"
    
    if [ -f "\$CELLPOSE_OUTPUT" ]; then
        mv "\$CELLPOSE_OUTPUT" "\$FINAL_OUTPUT"
        echo "Renamed: \$CELLPOSE_OUTPUT -> \$FINAL_OUTPUT"
        
        # Preserve metadata in segmentation mask
        python3 << 'PRESERVE_MASK_META'
import tifffile
import json
import numpy as np

# Load metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load mask
mask = tifffile.imread("\$FINAL_OUTPUT")

# Re-save with metadata
x_res = metadata['x_resolution_um'] / ${cfg.get('image_scaling', 1.0)}
y_res = metadata['y_resolution_um'] / ${cfg.get('image_scaling', 1.0)}
z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

tifffile.imwrite(
    "\$FINAL_OUTPUT",
    mask.astype(np.uint16),
    imagej=True,
    resolution=(1.0/x_res, 1.0/y_res),
    metadata={
        'spacing': z_spacing,
        'unit': 'um',
        'axes': 'ZYX',
        'TimePoint': metadata['imagej'].get('TimePoint', 0) if 'imagej' in metadata else 0,
        'LabelImage': True
    }
)
print(f"Metadata preserved in segmentation mask")
PRESERVE_MASK_META
        
    else
        echo "ERROR: Cellpose output not found: \$CELLPOSE_OUTPUT"
        ls -lh
        exit 1
    fi
    
    echo "Segmentation completed: \$FINAL_OUTPUT"
    """
}

// ============================================================================
// PROCESS: Merge Timepoints for Tracking
// ============================================================================

process MERGE_TIMEPOINTS {
    tag "${series_name}"
    
    publishDir "${params.output_dir}/02_segmented/timeseries", 
        mode: 'copy',
        pattern: "*_4D.tif"
    
    input:
    tuple val(series_name),
          path(segmented_files),
          path(metadata_json)
    
    output:
    tuple val(series_name),
          path("${series_name}_4D.tif"),
          path(metadata_json), emit: merged_4d
    
    script:
    """
    #!/usr/bin/env python3
    import tifffile
    import numpy as np
    import json
    from pathlib import Path
    import re
    
    print("Merging timepoints for: ${series_name}")
    
    # Load metadata
    with open('${metadata_json}', 'r') as f:
        metadata = json.load(f)
    
    # Get all segmented files and sort by timepoint
    seg_files = sorted(Path('.').glob('*_Cellseg.tif'))
    
    if not seg_files:
        raise ValueError("No segmented files found!")
    
    print(f"Found {len(seg_files)} timepoints")
    
    # Extract timepoint numbers and sort
    timepoint_files = []
    for f in seg_files:
        match = re.search(r't(\\d+)', f.name)
        if match:
            t = int(match.group(1))
            timepoint_files.append((t, f))
    
    timepoint_files.sort(key=lambda x: x[0])
    
    # Load all timepoints
    timepoint_arrays = []
    for t, f in timepoint_files:
        img = tifffile.imread(str(f))
        print(f"  Loaded t={t:04d}: shape={img.shape}, dtype={img.dtype}")
        timepoint_arrays.append(img)
    
    # Stack into 4D array (TZYX)
    img_4d = np.stack(timepoint_arrays, axis=0)
    print(f"\\nMerged 4D shape: {img_4d.shape}")
    
    # Calculate metadata
    x_res = metadata['x_resolution_um']
    y_res = metadata['y_resolution_um']
    z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0
    
    # Save 4D TIFF with full metadata for TrackMate
    tifffile.imwrite(
        '${series_name}_4D.tif',
        img_4d.astype(np.uint16),
        imagej=True,
        resolution=(1.0/x_res, 1.0/y_res),
        metadata={
            'spacing': z_spacing,
            'unit': 'um',
            'axes': 'TZYX',
            'frames': img_4d.shape[0],
            'slices': img_4d.shape[1],
            'LabelImage': True
        }
    )
    
    print(f"\\nSaved 4D timeseries: ${series_name}_4D.tif")
    print(f"  Shape: {img_4d.shape} (TZYX)")
    print(f"  Voxel size: {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} um")
    print(f"  Timepoints: {img_4d.shape[0]}")
    """
}

// ============================================================================
// PROCESS: TrackMate Tracking
// ============================================================================

process TRACKMATE_TRACK {
    tag "${series_name}"
    
    publishDir "${params.output_dir}/03_tracked", 
        mode: 'copy'
    
    publishDir "${params.output_dir}/logs/tracking",
        mode: 'copy',
        pattern: "*.log"
    
    container params.container
    
    input:
    tuple val(series_name),
          path(merged_4d),
          path(metadata_json)
    val tracking_config
    
    output:
    tuple val(series_name),
          path("${series_name}_tracks.xml"),
          path("${series_name}_tracks.csv"), emit: tracks
    path "${series_name}_tracking.log", emit: log
    
    script:
    def cfg = tracking_config
    """
    #!/usr/bin/env python3
    import json
    import sys
    import subprocess
    from pathlib import Path
    
    print("=" * 60)
    print(f"TrackMate Tracking: ${series_name}")
    print("=" * 60)
    
    # Load metadata
    with open('${metadata_json}', 'r') as f:
        metadata = json.load(f)
    
    # Load tracking config
    config = ${groovy.json.JsonOutput.toJson(cfg)}
    
    print("\\nTracking parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create TrackMate script
    trackmate_script = '''
import fiji.plugin.trackmate.Model
import fiji.plugin.trackmate.Settings
import fiji.plugin.trackmate.TrackMate
import fiji.plugin.trackmate.SelectionModel
import fiji.plugin.trackmate.Logger
import fiji.plugin.trackmate.detection.LabelImageDetectorFactory
import fiji.plugin.trackmate.tracking.jaqaman.SparseLAP JaqamanLinkerFactory
import fiji.plugin.trackmate.io.TmXmlWriter
import fiji.plugin.trackmate.action.ExportTracksToXML
import fiji.plugin.trackmate.action.ExportStatsToIJAction
import ij.IJ
import ij.ImagePlus

// Load image
imp = IJ.openImage("${merged_4d}")
imp.show()

// Create model
model = new Model()

// Create settings
settings = new Settings(imp)

// Configure Label Image Detector
settings.detectorFactory = new LabelImageDetectorFactory()
settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
settings.detectorSettings.put('SIMPLIFY_CONTOURS', ${cfg.simplify_contours})

// Configure tracker (Sparse LAP tracker for label images)
settings.trackerFactory = new SparseLAPJaqamanLinkerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
settings.trackerSettings.put('LINKING_MAX_DISTANCE', ${cfg.linking_max_distance})
settings.trackerSettings.put('GAP_CLOSING_MAX_DISTANCE', ${cfg.gap_closing_max_distance})
settings.trackerSettings.put('MAX_FRAME_GAP', ${cfg.max_frame_gap})
settings.trackerSettings.put('ALLOW_GAP_CLOSING', ${cfg.allow_gap_closing})
settings.trackerSettings.put('ALLOW_TRACK_SPLITTING', ${cfg.allow_track_splitting})
settings.trackerSettings.put('ALLOW_TRACK_MERGING', ${cfg.allow_track_merging})

// Run TrackMate
trackmate = new TrackMate(model, settings)

ok = trackmate.checkInput()
if (!ok) {
    println("Configuration error: " + trackmate.getErrorMessage())
    System.exit(1)
}

ok = trackmate.process()
if (!ok) {
    println("Processing error: " + trackmate.getErrorMessage())
    System.exit(1)
}

// Export results
// 1. Save TrackMate XML
outFile = new File("${series_name}_tracks.xml")
writer = new TmXmlWriter(outFile)
writer.appendModel(trackmate.getModel())
writer.appendSettings(trackmate.getSettings())
writer.writeToFile()
println("Saved TrackMate XML: ${series_name}_tracks.xml")

// 2. Export to CSV
model = trackmate.getModel()
spots = model.getSpots()
tracks = model.getTrackModel()

// Create CSV with track data
csvFile = new File("${series_name}_tracks.csv")
csvFile.withWriter { writer ->
    // Header
    writer.writeLine("TRACK_ID,SPOT_ID,FRAME,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,RADIUS,QUALITY")
    
    // Data
    trackIDs = tracks.trackIDs(true)
    trackIDs.each { trackID ->
        track = tracks.trackSpots(trackID)
        track.each { spot ->
            writer.writeLine([
                trackID,
                spot.ID(),
                spot.getFeature('FRAME'),
                spot.getFeature('POSITION_X'),
                spot.getFeature('POSITION_Y'),
                spot.getFeature('POSITION_Z'),
                spot.getFeature('POSITION_T'),
                spot.getFeature('RADIUS'),
                spot.getFeature('QUALITY')
            ].join(','))
        }
    }
}
println("Saved tracks CSV: ${series_name}_tracks.csv")

// Print summary
println("\\nTracking completed:")
println("  Total spots: " + model.getSpots().getNSpots(true))
println("  Total tracks: " + model.getTrackModel().nTracks(true))

System.exit(0)
'''
    
    # Save script
    with open('trackmate_script.groovy', 'w') as f:
        f.write(trackmate_script)
    
    # Run ImageJ/Fiji with TrackMate
    # Note: This requires ImageJ/Fiji to be available in the container
    cmd = [
        'fiji',
        '--headless',
        '--console',
        'trackmate_script.groovy'
    ]
    
    print("Running TrackMate...")
    print(f"Command: {' '.join(cmd)}\\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour timeout
    )
    
    # Save log
    with open('${series_name}_tracking.log', 'w') as f:
        f.write("STDOUT:\\n")
        f.write(result.stdout)
        f.write("\\n\\nSTDERR:\\n")
        f.write(result.stderr)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: TrackMate failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    # Validate outputs
    if not Path('${series_name}_tracks.xml').exists():
        print("ERROR: TrackMate XML not generated")
        sys.exit(1)
    
    if not Path('${series_name}_tracks.csv').exists():
        print("ERROR: Tracks CSV not generated")
        sys.exit(1)
    
    print("\\nTracking completed successfully!")
    """
}

// ============================================================================
// PROCESS: Generate QC Report
// ============================================================================

process GENERATE_QC_REPORT {
    publishDir "${params.output_dir}/reports", 
        mode: 'copy'
    
    input:
    path all_logs
    path config_json
    
    output:
    path "pipeline_report.html"
    path "pipeline_summary.json"
    
    script:
    """
    #!/usr/bin/env python3
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Collect all log files
    log_files = list(Path('.').glob('*.log'))
    
    # Load config
    with open('${config_json}', 'r') as f:
        config = json.load(f)
    
    # Generate summary
    summary = {
        'pipeline_version': '1.0.0',
        'execution_date': datetime.now().isoformat(),
        'configuration': config,
        'processed_images': len([f for f in log_files if 'preprocess' in f.name]),
        'segmented_images': len([f for f in log_files if 'segment' in f.name]),
        'tracked_series': len([f for f in log_files if 'tracking' in f.name])
    }
    
    # Save summary JSON
    with open('pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate HTML report
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>SPIM Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .info {{ color: #3498db; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>SPIM 4D Image Processing Pipeline Report</h1>
    
    <h2>Execution Summary</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Pipeline Version</td><td>{summary['pipeline_version']}</td></tr>
        <tr><td>Execution Date</td><td>{summary['execution_date']}</td></tr>
        <tr><td>Images Preprocessed</td><td class="success">{summary['processed_images']}</td></tr>
        <tr><td>Images Segmented</td><td class="success">{summary['segmented_images']}</td></tr>
        <tr><td>Series Tracked</td><td class="success">{summary['tracked_series']}</td></tr>
    </table>
    
    <h2>Configuration</h2>
    <pre>{json.dumps(config, indent=2)}</pre>
    
    <h2>Processing Steps</h2>
    <ol>
        <li><strong>Metadata Extraction</strong> - Preserved original image metadata</li>
        <li><strong>Timepoint Splitting</strong> - Split 4D images into individual 3D timepoints</li>
        <li><strong>Preprocessing & Deconvolution</strong> - Applied corrections and deconvolution</li>
        <li><strong>Cellpose Segmentation</strong> - Detected and labeled cells</li>
        <li><strong>Timepoint Merging</strong> - Reassembled 4D timeseries</li>
        <li><strong>TrackMate Tracking</strong> - Tracked cells across time</li>
    </ol>
    
    <p><em>Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
</body>
</html>
'''
    
    with open('pipeline_report.html', 'w') as f:
        f.write(html)
    
    print("QC report generated successfully")
    """
}

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

workflow {
    // Input channel: all TIFF files in input directory
    input_images = Channel
        .fromPath("${params.input_dir}/*.{tif,tiff}")
        .ifEmpty { error "No TIFF files found in ${params.input_dir}" }
    
    // 1. Extract metadata from each image
    EXTRACT_METADATA(input_images)
    
    // 2. Split 4D images into timepoints
    SPLIT_TIMEPOINTS(EXTRACT_METADATA.out.with_metadata)
    
    // Flatten timepoints for parallel processing
    timepoints_flat = SPLIT_TIMEPOINTS.out.timepoints
        .transpose()
        .map { series_name, timepoint_file, metadata -> 
            tuple(series_name, timepoint_file, metadata)
        }
    
    // 3. Preprocess and deconvolve each timepoint
    PREPROCESS_DECONVOLVE(
        timepoints_flat,
        config.preprocessing
    )
    
    // 4. Segment each timepoint
    CELLPOSE_SEGMENT(
        PREPROCESS_DECONVOLVE.out.processed,
        config.segmentation
    )
    
    // 5. Group timepoints by series and merge
    segmented_grouped = CELLPOSE_SEGMENT.out.segmented
        .groupTuple(by: 0)
        .map { series_name, files, metadata ->
            tuple(series_name, files, metadata[0])
        }
    
    MERGE_TIMEPOINTS(segmented_grouped)
    
    // 6. Track cells across time
    TRACKMATE_TRACK(
        MERGE_TIMEPOINTS.out.merged_4d,
        config.tracking
    )
    
    // 7. Generate QC report
    all_logs = PREPROCESS_DECONVOLVE.out.log
        .mix(CELLPOSE_SEGMENT.out.log)
        .mix(TRACKMATE_TRACK.out.log)
        .collect()
    
    GENERATE_QC_REPORT(
        all_logs,
        params.config_json
    )
}

// ============================================================================
// WORKFLOW COMPLETION
// ============================================================================

workflow.onComplete {
    log.info """
    ============================================================================
    Pipeline completed!
    ============================================================================
    Status      : ${workflow.success ? 'SUCCESS' : 'FAILED'}
    Duration    : ${workflow.duration}
    Output dir  : ${params.output_dir}
    
    Results:
      - Preprocessed images : ${params.output_dir}/01_preprocessed/
      - Segmented masks     : ${params.output_dir}/02_segmented/
      - Tracking results    : ${params.output_dir}/03_tracked/
      - QC reports          : ${params.output_dir}/reports/
      - Logs                : ${params.output_dir}/logs/
    
    Completed at: ${workflow.complete}
    ============================================================================
    """.stripIndent()
}

workflow.onError {
    log.error """
    ============================================================================
    Pipeline execution failed!
    ============================================================================
    Error message: ${workflow.errorMessage}
    Error report : ${workflow.errorReport}
    ============================================================================
    """.stripIndent()
}

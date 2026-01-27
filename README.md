# SPIM 4D Image Processing Pipeline

A comprehensive Nextflow pipeline for processing 4D SPIM (Selective Plane Illumination Microscopy) images through preprocessing, deconvolution, segmentation, and tracking.

## Features

- **Metadata Preservation**: Maintains voxel size and imaging metadata throughout the pipeline for TrackMate compatibility
- **Parallel Processing**: Processes timepoints in parallel for optimal performance on SLURM clusters
- **Reproducibility**: All parameters stored in JSON configuration files
- **Quality Control**: Automatic generation of execution reports and visualizations
- **End-to-End**: From raw 4D images to cell tracking results

## Pipeline Overview

```
Raw 4D TIFF Images
    ↓
1. Extract Metadata
    ↓
2. Split into Timepoints (3D stacks)
    ↓
3. Preprocess & Deconvolve (parallel)
    ├─ Shading correction
    ├─ Z-intensity correction
    ├─ Isotropic reslicing
    ├─ 3D deconvolution (GPU)
    ├─ Background subtraction
    └─ CLAHE enhancement
    ↓
4. Cellpose Segmentation (parallel)
    └─ 3D cell detection with custom models
    ↓
5. Merge Timepoints → 4D Timeseries
    ↓
6. TrackMate Tracking
    └─ Cell tracking across time
    ↓
Output: Tracks (XML + CSV) + QC Reports
```

## Requirements

### Software

- Nextflow ≥ 23.04.0
- Singularity/Apptainer (for SLURM) or Docker (for local)
- Java ≥ 11 (for Nextflow)

### Container

The pipeline uses a pre-built container with all dependencies:
- Python 3.9+ with scientific libraries
- CUDA support for GPU acceleration
- Cellpose
- RedLionfish deconvolution
- ImageJ/Fiji with TrackMate

Container: `docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-8720eea`

### Hardware

**Recommended per job:**
- Preprocessing/Deconvolution: 8 CPUs, 64 GB RAM, 1 GPU
- Segmentation: 4 CPUs, 64 GB RAM, 1 GPU
- Tracking: 4 CPUs, 32 GB RAM

## Installation

### 1. Install Nextflow - Already installed in the CLIP Cluster

```bash
# Download Nextflow
curl -s https://get.nextflow.io | bash

# Move to PATH
sudo mv nextflow /usr/local/bin/

# Verify installation
nextflow -version
```

### 2. Clone Pipeline

```bash
git clone https://github.com/your-repo/spim-pipeline.git
cd spim-pipeline
```

### 3. Test Installation

```bash
# Test on a small dataset
nextflow run spim_pipeline.nf \
    --input_dir test_data \
    --output_dir test_output \
    --config_json test_config.json \
    -profile local
```

## Quick Start

### 1. Prepare Your Data

Organize your raw 4D TIFF images in a directory:

```
raw_data/
├── embryo_01.tif  (4D: TZYX)
├── embryo_02.tif
└── embryo_03.tif
```

### 2. Create Configuration File

Copy and edit the configuration template:

```bash
cp config_template.json my_experiment.json
nano my_experiment.json
```

Key parameters to configure:
- `preprocessing.psf_path`: Path to your PSF model
- `segmentation.model`: Path to your Cellpose model
- `segmentation.diameter`: Expected cell diameter in pixels

### 3. Run the Pipeline

**On SLURM cluster:**

```bash
nextflow run spim_pipeline.nf \
    --input_dir /path/to/raw_data \
    --output_dir /path/to/results \
    --config_json my_experiment.json \
    -profile standard \
    -resume
```

**For high-resolution images:**

```bash
nextflow run spim_pipeline.nf \
    --input_dir /path/to/raw_data \
    --output_dir /path/to/results \
    --config_json my_experiment.json \
    -profile highres \
    -resume
```

**Local testing:**

```bash
nextflow run spim_pipeline.nf \
    --input_dir ./test_data \
    --output_dir ./test_output \
    --config_json test_config.json \
    -profile local
```

## Output Structure

```
output_dir/
├── 01_preprocessed/
│   ├── timepoints/          # Individual 3D timepoints
│   │   ├── embryo_01_t0000.tif
│   │   ├── embryo_01_t0001.tif
│   │   └── ...
│   └── processed/           # Preprocessed timepoints
│       ├── embryo_01_t0000_processed.tif
│       └── ...
│
├── 02_segmented/
│   ├── embryo_01_t0000_diam30_Cellseg.tif
│   ├── ...
│   └── timeseries/          # Merged 4D segmentation
│       └── embryo_01_4D.tif
│
├── 03_tracked/
│   ├── embryo_01_tracks.xml      # TrackMate XML
│   ├── embryo_01_tracks.csv      # Track data
│   └── ...
│
├── metadata/
│   ├── embryo_01_metadata.json   # Original metadata
│   └── ...
│
├── logs/
│   ├── preprocessing/
│   ├── segmentation/
│   └── tracking/
│
└── reports/
    ├── pipeline_report.html      # QC report
    ├── execution_report.html     # Nextflow report
    ├── timeline.html             # Execution timeline
    ├── trace.txt                 # Resource usage
    └── pipeline_dag.html         # Pipeline DAG
```

## Configuration Reference

### Preprocessing Parameters

```json
{
  "preprocessing": {
    "psf_path": "/path/to/PSF.tif",
    "image_scaling": 0.5,              // XY downsampling (0.5 = 50%)
    "deconvolution": {
      "niter": 3,                      // 3D deconvolution iterations
      "niterz": 3,                     // 2D XZ iterations
      "padding": 32                    // Edge padding
    },
    "normalization": {
      "percentile_low": 40.0,          // Lower intensity cutoff
      "percentile_high": 99.99         // Upper intensity cutoff
    },
    "background_subtraction": {
      "resolution_px0": 10,            // XY resolution (microns)
      "resolution_pz0": 10,            // Z resolution (microns)
      "noise_lvl": 2                   // Noise level (integer)
    },
    "postprocessing": {
      "sigma": 1.0                     // Gaussian smoothing
    },
    "correction_flags": {
      "no_clahe": false,               // Disable CLAHE
      "no_z_correction": false,        // Disable Z correction
      "no_shading": false              // Disable shading correction
    }
  }
}
```

### Segmentation Parameters

```json
{
  "segmentation": {
    "model": "/path/to/cellpose/model",
    "diameter": 30,                    // Cell diameter in pixels
    "flow_threshold": 0.8,             // 0-3, higher = more conservative
    "cellprob_threshold": 0.0,         // -6 to 6, higher = fewer cells
    "use_gpu": true,
    "do_3d": true,                     // Required for 3D segmentation
    "save_tif": true,                  // Required for 3D
    "save_flows": false,
    "save_npy": false
  }
}
```

### Tracking Parameters

```json
{
  "tracking": {
    "linking": {
      "linking_max_distance": 15.0    // Frame-to-frame linking distance
    },
    "gap_closing": {
      "allow_gap_closing": true,
      "gap_closing_max_distance": 15.0,
      "max_frame_gap": 2               // Max frames to bridge
    },
    "splitting_merging": {
      "allow_track_splitting": false,  // Enable for cell division
      "allow_track_merging": false     // Enable for cell fusion
    }
  }
}
```

## Advanced Usage

### Resume Failed Runs

Nextflow automatically resumes from the last successful step:

```bash
nextflow run spim_pipeline.nf \
    --input_dir /path/to/data \
    --output_dir /path/to/results \
    --config_json config.json \
    -resume
```

### Process Specific Files

Edit the pipeline to filter input files:

```groovy
// In spim_pipeline.nf, modify the input channel:
input_images = Channel
    .fromPath("${params.input_dir}/*_embryo_*.tif")
    .filter { it.name.contains('Series01') }
```

### Adjust Resource Allocation

Edit `nextflow.config` to modify resources per process:

```groovy
process {
    withName: 'PREPROCESS_DECONVOLVE' {
        cpus = 16
        memory = 128.GB
        time = 6.h
    }
}
```

### Clean Up Work Directory

After successful completion:

```bash
# Remove intermediate files
nextflow clean -f

# Or configure automatic cleanup in nextflow.config
cleanup = true
```

## Troubleshooting

### Issue: GPU Not Detected

**Solution:** Verify GPU access in container:

```bash
singularity exec --nv docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-8720eea \
    nvidia-smi
```

### Issue: Out of Memory

**Solutions:**
1. Increase memory allocation in `nextflow.config`
2. Use the `highres` profile
3. Increase `image_scaling` (more downsampling)
4. Process fewer timepoints in parallel

### Issue: Metadata Not Preserved

**Check:**
1. Original TIFF contains metadata: `tiffinfo input.tif`
2. Metadata JSON generated: `ls output_dir/metadata/`
3. Processed images have metadata: `tiffinfo processed.tif`

### Issue: TrackMate Fails

**Common causes:**
1. Missing ImageJ/Fiji in container
2. 4D image not properly formatted (check axes: TZYX)
3. Insufficient memory for large timeseries

**Solution:** Check tracking log:
```bash
cat output_dir/logs/tracking/*_tracking.log
```

## Performance Optimization

### Parallel Processing

The pipeline automatically parallelizes across:
- Multiple input images
- Timepoints within each image

For 10 images with 50 timepoints each:
- Sequential: ~500 jobs
- Parallel: Limited by SLURM queue size

### Resource Guidelines

| Image Size | Profile | Memory | Time |
|------------|---------|--------|------|
| 512x512x100 | standard | 64 GB | 2h |
| 1024x1024x200 | highres | 128 GB | 6h |
| 2048x2048x300 | custom | 256 GB | 12h |

### Optimization Tips

1. **Use appropriate downsampling**: `image_scaling: 0.5` reduces memory by 75%
2. **Skip unnecessary steps**: Set flags in config (`no_clahe: true`)
3. **Adjust deconvolution iterations**: Fewer iterations = faster processing
4. **Pre-filter input files**: Process only required images

## Citation

If you use this pipeline, please cite:

```bibtex
@software{spim_pipeline,
  title = {SPIM 4D Image Processing Pipeline},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/your-repo/spim-pipeline}
}
```

And the underlying tools:
- **Cellpose**: Stringer et al., Nature Methods 2021
- **TrackMate**: Tinevez et al., Methods 2017
- **RedLionfish**: [Citation if available]

## License

[Specify your license]

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-repo/spim-pipeline/issues
- Email: your.email@institution.edu

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Changelog

### Version 1.0.0 (2026-01-27)
- Initial release
- Metadata preservation throughout pipeline
- SLURM cluster support
- Comprehensive QC reporting
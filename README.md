# SPIM 4D Image Processing Pipeline - Corrected Version

## 🔧 Key Changes from Original Pipeline

### 1. **File Naming Pattern Recognition**
The corrected pipeline properly handles the specific naming convention:
- **Pattern**: `t0051_Channel 1.tif`, `t0052_Channel 1.tif`, etc.
- **Features**:
  - Extracts timepoint number from `t####` prefix
  - Handles the space in "Channel #" correctly
  - Filters by channel number (default: 1, configurable via `--channel`)
  - Sorts timepoints numerically for correct ordering

### 2. **Simplified Workflow**
The pipeline now follows your actual requirements:

```
Input: Individual 3D Z-stacks per timepoint
  ↓
Step 1: Extract metadata from each timepoint
  ↓
Step 2: Preprocess & deconvolve each timepoint
  ↓
Step 3: Cellpose segmentation on each timepoint
  ↓
Step 4: Merge all timepoints into single 4D hyperstack
  ↓
Output: 4D_hyperstack.tif (TZYX with preserved metadata)
```

### 3. **Removed Unnecessary Steps**
- ❌ Removed: Initial 4D splitting (files are already individual timepoints)
- ❌ Removed: TrackMate tracking (can be done separately on the final hyperstack)
- ✅ Kept: Essential processing and metadata preservation

### 4. **Metadata Preservation**
- Extracts and preserves voxel size from original images
- Accounts for scaling during preprocessing
- Properly sets ImageJ metadata in final hyperstack:
  - Axes order: TZYX
  - Frame count
  - Slice count
  - Resolution (X, Y)
  - Z-spacing

## 📋 Usage

### Basic Command
```bash
nextflow run spim_pipeline_fixed.nf \
    --input_dir /path/to/timepoint/images \
    --output_dir /path/to/output \
    --config_json config.json \
    --channel 1
```

### Parameters
- `--input_dir`: Directory containing `t####_Channel #.tif` files
- `--output_dir`: Where to save all outputs
- `--config_json`: Configuration file (see example below)
- `--channel`: Channel number to process (default: 1)
- `--container`: Container image (has default value)

### Example Input Directory Structure
```
input_dir/
├── t0001_Channel 1.tif  ← Z-stack, timepoint 1
├── t0002_Channel 1.tif  ← Z-stack, timepoint 2
├── t0003_Channel 1.tif
├── t0004_Channel 1.tif
└── ...
├── t0001_Channel 2.tif  ← Will be ignored if --channel 1
└── ...
```

### Output Directory Structure
```
output_dir/
├── 01_preprocessed/
│   ├── t0001_processed.tif
│   ├── t0002_processed.tif
│   └── ...
├── 02_segmented/
│   ├── t0001_segmented.tif
│   ├── t0002_segmented.tif
│   └── ...
├── 03_hyperstack/
│   ├── 4D_hyperstack.tif          ← FINAL OUTPUT
│   └── 4D_hyperstack_metadata.json
├── metadata/
│   ├── t0001_metadata.json
│   ├── t0002_metadata.json
│   └── ...
├── logs/
│   ├── preprocessing/
│   └── segmentation/
└── reports/
    ├── pipeline_report.html
    └── pipeline_summary.json
```

## ⚙️ Configuration File

The configuration JSON has been updated to match the actual structure used in the code:

```json
{
  "pipeline_info": {
    "version": "1.0.0",
    "description": "SPIM Pipeline Configuration",
    "experiment": "Your experiment name"
  },

  "preprocessing": {
    "psf_path": "/path/to/PSF.tif",
    "image_scaling": 0.5,

    "deconvolution": {
      "niter": 6,
      "niterz": 6,
      "padding": 32
    },

    "normalization": {
      "min_v": 0,
      "max_v": 65535,
      "percentile_low": 40.0,
      "percentile_high": 99.99
    },

    "background_subtraction": {
      "resolution_px0": 10,
      "resolution_pz0": 10,
      "noise_lvl": 2
    },

    "postprocessing": {
      "sigma": 1.0
    },

    "correction_flags": {
      "no_clahe": false,
      "no_z_correction": false,
      "no_shading": false
    }
  },

  "segmentation": {
    "model": "/path/to/cellpose/model",
    "diameter": 30,
    "flow_threshold": 0.8,
    "cellprob_threshold": 0.0,
    "use_gpu": true,
    "do_3d": true,
    "save_tif": true,
    "save_flows": false,
    "save_npy": false,
    "image_scaling": 0.5
  }
}
```

### Critical Configuration Notes

1. **`image_scaling`**: Must be the same in both preprocessing and segmentation sections
   - This ensures voxel sizes are calculated correctly
   - Example: 0.5 = 50% downsampling

2. **`diameter`**: Cellpose cell diameter in pixels
   - ⚠️ **NEVER use 0 for 3D segmentation** (will cause errors)
   - Should match expected cell size after scaling
   - Example: 30 pixels ≈ 15 µm diameter cells at 0.5 µm/pixel

3. **`do_3d`**: Must be `true` for volumetric Z-stacks

## 🔍 Key Differences from Original

| Aspect | Original | Corrected |
|--------|----------|-----------|
| Input assumption | Single 4D TIFF | Multiple 3D TIFFs (one per timepoint) |
| File parsing | Generic pattern | Specific `t####_Channel #.tif` pattern |
| Channel selection | Not available | `--channel` parameter |
| Workflow | Split → Process → Merge → Track | Process → Merge |
| Output | Tracked XML + CSV | 4D hyperstack TIFF |
| Tracking | Built-in TrackMate | Separate step (on final hyperstack) |

## 🎯 Use Cases

### After Pipeline Completion

The final `4D_hyperstack.tif` can be used for:

1. **TrackMate tracking** (if needed):
   - Open in Fiji/ImageJ
   - Plugins → Tracking → TrackMate
   - Use "Label image detector" since it's already segmented

2. **Direct visualization**:
   - Open in Fiji/ImageJ
   - Already has correct voxel size metadata
   - Can view as hyperstack (Image → Hyperstacks → Stack to Hyperstack)

3. **Further analysis**:
   - Load in Python with tifffile
   - Analyze cell volumes, shapes, movements
   - Export to other formats

## 🐛 Troubleshooting

### Problem: "No files found"
**Solution**: Check that:
- Files are named exactly as `t####_Channel #.tif`
- There's a space between "Channel" and the number
- The channel number matches your `--channel` parameter

### Problem: "Could not parse timepoint"
**Solution**: Verify filename format. The `t####` must be:
- Lowercase 't'
- Followed by digits
- Example: `t0001`, `t0051`, `t0123`

### Problem: Cellpose "diameter cannot be 0"
**Solution**: In config.json, set a specific diameter value:
```json
"diameter": 30  // NOT 0 for 3D!
```

### Problem: Inconsistent voxel sizes
**Solution**: Ensure `image_scaling` is identical in both sections:
```json
"preprocessing": {
  "image_scaling": 0.5
},
"segmentation": {
  "image_scaling": 0.5  // Must match!
}
```

## 📊 Expected Timeline

For a typical dataset (50 timepoints, 512×512×100 per timepoint):

| Step | Time per Timepoint | Total Time |
|------|-------------------|------------|
| Metadata extraction | < 1 min | < 1 hour |
| Preprocessing | 5-10 min | 4-8 hours |
| Segmentation | 2-5 min | 2-4 hours |
| Merging | N/A | < 5 min |
| **Total** | | **~6-13 hours** |

Times depend on:
- GPU availability
- Image size
- Number of iterations
- Cluster load

## 🚀 Running on SLURM Cluster

Create a `nextflow.config` file:

```groovy
process {
    executor = 'slurm'
    queue = 'gpu'

    withName: PREPROCESS_DECONVOLVE {
        cpus = 8
        memory = '64 GB'
        time = '2h'
        clusterOptions = '--gres=gpu:1'
    }

    withName: CELLPOSE_SEGMENT {
        cpus = 4
        memory = '64 GB'
        time = '2h'
        clusterOptions = '--gres=gpu:1 --exclude=clip-g1-[0-6]'
    }

    withName: MERGE_TO_HYPERSTACK {
        cpus = 4
        memory = '32 GB'
        time = '30m'
    }
}

singularity {
    enabled = true
    autoMounts = true
}
```

Then run:
```bash
nextflow run spim_pipeline_fixed.nf \
    --input_dir /groups/yourlab/data/raw \
    --output_dir /groups/yourlab/data/processed \
    --config_json config.json \
    --channel 1 \
    -c nextflow.config
```

## 📝 Notes

1. **Parallelization**: Each timepoint is processed independently, so the pipeline automatically parallelizes across available resources.

2. **Resume capability**: Nextflow supports `-resume` to continue from where it stopped if interrupted.

3. **Memory requirements**: Adjust based on your image sizes. Larger images need more memory.

4. **GPU requirements**: Both preprocessing (deconvolution) and segmentation benefit from GPU acceleration.

## 📚 Related Files

- `spim_pipeline_fixed.nf` - Main Nextflow pipeline
- `config.json` - Configuration parameters
- `spim_pipeline_fixed.py` - Preprocessing script (assumed to exist in container)

## 🔗 Dependencies

The container should include:
- Python 3.8+
- tifffile
- numpy
- cellpose
- Preprocessing script (`spim_pipeline_fixed.py`)

All handled by the specified container image.